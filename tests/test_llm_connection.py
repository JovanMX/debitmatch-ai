"""Smoke test: verify the OpenAI LLM configured in config.py / .env is reachable.

Run with:  python tests/test_llm_connection.py
Exit codes: 0 = operational, 1 = configuration problem, 2 = API/connection error.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/ sits one level below the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# config.py messages contain emoji; avoid UnicodeEncodeError on legacy Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PLACEHOLDER_KEY = "COLOCAR-AQUI-APIKEY"
TEST_PROMPT = "Reply with exactly one short sentence confirming you are online."


def main() -> int:
    try:
        from config import TASK_MODEL_DEFAULTS, get_task_llm, task_model
        import openai
    except ImportError as exc:
        print(f"[ERROR] Missing dependency: {exc}. Run: pip install langchain-openai python-dotenv")
        return 1

    checked = set()
    for task in TASK_MODEL_DEFAULTS:  # every distinct per-task model (profiling, advisory, critic, judge, ...)
        spec = task_model(task)
        if (spec.model, spec.reasoning_effort) in checked:
            continue
        checked.add((spec.model, spec.reasoning_effort))
        code = _check(task, spec, get_task_llm, openai)
        if code:
            return code
    return 0


def _check(task, spec, get_task_llm, openai) -> int:
    try:
        llm = get_task_llm(task)
    except ValueError as exc:
        print("[CONFIG] The OpenAI API key is not configured.")
        print(f"         {exc}")
        print(f"         Replace '{PLACEHOLDER_KEY}' in the .env file with your real key: OPENAI_API_KEY=sk-...")
        return 1

    model_name = spec.model + (f" (reasoning_effort={spec.reasoning_effort})" if spec.reasoning_effort else "")
    print(f"[{task}] Invoking {model_name} ...")
    started = time.perf_counter()
    try:
        response = llm.invoke(TEST_PROMPT)
    except openai.AuthenticationError:
        print("[AUTH] OpenAI rejected the API key (401). Check OPENAI_API_KEY in .env.")
        return 1
    except openai.RateLimitError as exc:
        print(f"[QUOTA] Rate limit or insufficient quota (429): {exc}")
        return 2
    except openai.NotFoundError:
        print(f"[MODEL] Model '{model_name}' not found or not available for this key.")
        return 1
    except openai.APIConnectionError as exc:
        print(f"[NETWORK] Could not reach the OpenAI API: {exc}")
        return 2
    except openai.APIError as exc:
        print(f"[API] OpenAI API error: {exc}")
        return 2

    elapsed = time.perf_counter() - started
    usage = getattr(response, "usage_metadata", None) or {}
    print(f"Model response: {response.content}")
    print(f"Latency: {elapsed:.2f}s | tokens in/out: {usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}")
    print(f"[OK] {model_name} is operational in DebitMatch AI.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
