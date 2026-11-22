"""Structured (JSON) system prompts and LLM output contracts for each agent."""

from prompts.advisory_prompts import ADVISORY_SYSTEM_PROMPT, AdvisoryNarrative
from prompts.evaluator_prompts import EVALUATOR_CRITIC_SYSTEM_PROMPT, CriticVerdict
from prompts.profiling_prompts import ARCHETYPES, PROFILING_SYSTEM_PROMPT, ProfilingLLMOutput

__all__ = [
    "ADVISORY_SYSTEM_PROMPT",
    "ARCHETYPES",
    "AdvisoryNarrative",
    "CriticVerdict",
    "EVALUATOR_CRITIC_SYSTEM_PROMPT",
    "PROFILING_SYSTEM_PROMPT",
    "ProfilingLLMOutput",
]
