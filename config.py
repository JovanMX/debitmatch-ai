import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load variables from the .env file
load_dotenv()

logger = logging.getLogger(__name__)

PLACEHOLDER_API_KEY = "COLOCAR-AQUI-APIKEY"
DEFAULT_LANGSMITH_PROJECT = "debitmatch-ai"

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1  # non-reasoning models only: near-deterministic weights for reproducible runs
# Reasoning models only accept the default temperature (1); variability is set with reasoning_effort.
REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})


def is_reasoning_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith(REASONING_MODEL_PREFIXES)


@dataclass(frozen=True)
class TaskModel:
    """Model used by one LLM task. ``reasoning_effort`` applies to reasoning models only."""

    model: str
    reasoning_effort: Optional[str] = None


# Per-task routing (measured on recorded graph calls):
#   profiling / memory -> gpt-4o-mini (fastest on short classification prompts)
#   advisory / critic  -> gpt-6-luna, reasoning_effort="none" (faster and more faithful than gpt-4o-mini)
#   judge              -> gpt-6.1-terra (benchmark judge differs from the model that writes the narrative)
# Override with DEBITMATCH_MODEL_<TASK> / DEBITMATCH_REASONING_EFFORT_<TASK> in .env.
TASK_MODEL_DEFAULTS = {
    "profiling": TaskModel(DEFAULT_CHAT_MODEL),
    "memory": TaskModel(DEFAULT_CHAT_MODEL),
    "advisory": TaskModel("gpt-6-luna", "none"),
    "critic": TaskModel("gpt-6-luna", "none"),
    "judge": TaskModel("gpt-6.1-terra"),
}
DEFAULT_TASK = "profiling"


def task_model(task: str) -> TaskModel:
    """Resolve the model of ``task`` from the .env overrides or ``TASK_MODEL_DEFAULTS``."""
    default = TASK_MODEL_DEFAULTS.get(task, TASK_MODEL_DEFAULTS[DEFAULT_TASK])
    suffix = task.upper()
    model = os.getenv(f"DEBITMATCH_MODEL_{suffix}")
    if not model and task in {"profiling", "memory"}:
        model = os.getenv("OPENAI_MODEL_NAME")  # legacy single-model setting
    effort = os.getenv(f"DEBITMATCH_REASONING_EFFORT_{suffix}", default.reasoning_effort or "") or None
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError(f"DEBITMATCH_REASONING_EFFORT_{suffix}={effort!r}; expected one of {sorted(REASONING_EFFORTS)}.")
    return TaskModel((model or default.model).strip(), effort)


def get_llm(
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> ChatOpenAI:
    """
    Build the ChatOpenAI client configured safely from .env.

    Args:
        temperature: Overrides OPENAI_TEMPERATURE (non-reasoning models only; reasoning
            models reject any value other than their default, so it is never sent to them).
        timeout: Per-request timeout in seconds (used to honour the squad's
            execution budget).
        max_retries: Overrides the OpenAI client retry count.
        model: Overrides OPENAI_MODEL_NAME (e.g. a different judge model for evaluation).
        reasoning_effort: Reasoning models only ("none" ... "xhigh"); ignored otherwise.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = model or os.getenv("OPENAI_MODEL_NAME", DEFAULT_CHAT_MODEL)

    if not api_key or api_key == PLACEHOLDER_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured correctly in the .env file. "
            "Edit .env and set your real API key."
        )

    optional_kwargs = {}
    if timeout is not None:
        optional_kwargs["timeout"] = timeout
    if max_retries is not None:
        optional_kwargs["max_retries"] = max_retries
    if is_reasoning_model(model_name):
        if reasoning_effort:
            optional_kwargs["reasoning_effort"] = reasoning_effort
    else:
        optional_kwargs["temperature"] = (
            temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE)))

    return ChatOpenAI(model=model_name, api_key=api_key, **optional_kwargs)


def get_task_llm(task: str, timeout: Optional[float] = None, max_retries: Optional[int] = None,
                 temperature: Optional[float] = None) -> ChatOpenAI:
    """ChatOpenAI client for one LLM task (see ``TASK_MODEL_DEFAULTS``)."""
    spec = task_model(task)
    return get_llm(temperature=temperature, timeout=timeout, max_retries=max_retries,
                   model=spec.model, reasoning_effort=spec.reasoning_effort)


def get_langsmith_settings(project: Optional[str] = None) -> dict:
    """LangSmith client settings read from .env (LANGCHAIN_* or LANGSMITH_* variables).

    Returns a dict with ``api_key`` (None when not configured), ``project``,
    ``endpoint`` and ``enabled``. The key itself is never logged.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or None
    if api_key is not None and not api_key.strip():
        api_key = None
    resolved_project = (project or os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
                        or DEFAULT_LANGSMITH_PROJECT)
    endpoint = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    return {"api_key": api_key, "project": resolved_project, "endpoint": endpoint, "enabled": api_key is not None}


def configure_tracing(project: Optional[str] = None) -> bool:
    """Enable LangSmith tracing only when it is requested AND credentials exist.

    LangGraph / LangChain trace natively once these env vars are set. Tracing
    requested without an API key is switched off to avoid noisy 401 warnings.

    Returns:
        True when traces will be sent to LangSmith.
    """
    requested = any(
        os.getenv(var, "false").strip().lower() == "true"
        for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    if requested and not api_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning("LangSmith tracing requested but no LANGCHAIN_API_KEY/LANGSMITH_API_KEY set; tracing disabled.")
        return False

    if requested:
        resolved_project = project or os.getenv("LANGCHAIN_PROJECT") or DEFAULT_LANGSMITH_PROJECT
        os.environ["LANGCHAIN_PROJECT"] = resolved_project
        os.environ["LANGSMITH_PROJECT"] = resolved_project
        os.environ["LANGSMITH_TRACING"] = "true"
    return requested
