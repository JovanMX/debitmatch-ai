"""LLM abstraction: structured (Pydantic) generation with a hard per-call timeout.

Agents depend on the :class:`StructuredLLM` protocol, not on OpenAI directly,
so tests can inject deterministic fakes and production can swap providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be called (no key, no time budget, API failure)."""


@runtime_checkable
class StructuredLLM(Protocol):
    model_name: str

    def generate(
        self,
        schema: Type[T],
        system_prompt: str,
        payload: Mapping[str, Any],
        timeout_s: float,
        run_name: str,
    ) -> T: ...


# run_name of each agent call -> task of ``config.TASK_MODEL_DEFAULTS`` (one model per task).
TASK_BY_RUN_NAME = {
    "profiling_llm": "profiling",
    "profiling_extraction_llm": "profiling",  # hybrid parameter extraction (agents/extraction.py)
    "memory_summarizer_llm": "memory",
    "advisory_llm": "advisory",
    "evaluator_critic_llm": "critic",
}


class OpenAIStructuredLLM:
    """Per-task OpenAI models via ``config.get_task_llm`` + structured output.

    The task is resolved from ``run_name`` (``TASK_BY_RUN_NAME``); unknown run names
    use the profiling model.
    """

    def __init__(self, temperature: Optional[float] = None) -> None:
        from config import get_task_llm, task_model  # local import: keeps tools usable without LangChain

        self._get_task_llm = get_task_llm
        self._temperature = temperature
        get_task_llm("profiling", temperature=temperature)  # fail fast on a missing/placeholder key
        self.task_models = {task: task_model(task) for task in sorted(set(TASK_BY_RUN_NAME.values()))}
        self.model_name = ", ".join(
            f"{task}={spec.model}" + (f"({spec.reasoning_effort})" if spec.reasoning_effort else "")
            for task, spec in self.task_models.items())

    def model_for(self, run_name: str) -> str:
        return self.task_models[TASK_BY_RUN_NAME.get(run_name, "profiling")].model

    def generate(
        self,
        schema: Type[T],
        system_prompt: str,
        payload: Mapping[str, Any],
        timeout_s: float,
        run_name: str,
    ) -> T:
        from langchain_core.messages import HumanMessage, SystemMessage

        if timeout_s <= 0:
            raise LLMUnavailableError("No time budget left for an LLM call.")
        llm = self._get_task_llm(TASK_BY_RUN_NAME.get(run_name, "profiling"), timeout=timeout_s, max_retries=0,
                                 temperature=self._temperature)
        runnable = llm.with_structured_output(schema).with_config(run_name=run_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        try:
            result = runnable.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - provider errors vary widely; normalise them
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        return result if isinstance(result, schema) else schema.model_validate(result)


def build_default_llm() -> Optional[StructuredLLM]:
    """Return the OpenAI client, or None (deterministic fallback mode) if not configured."""
    try:
        return OpenAIStructuredLLM()
    except (ValueError, ImportError) as exc:
        logger.warning("LLM disabled, running in deterministic fallback mode: %s", exc)
        return None
