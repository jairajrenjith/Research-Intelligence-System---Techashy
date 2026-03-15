"""
core/llm_router.py
------------------
Central LLM router using LiteLLM.
- Each agent type has a prioritised provider list
- Auto-rotates on 429 / timeout
- Zero code change needed when adding new providers
"""

import os
import asyncio
import logging
from typing import Optional
from litellm import acompletion, completion
from litellm.exceptions import RateLimitError, APIConnectionError, Timeout
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Provider pools per agent ──────────────────────────────────────────────────
# Order = priority. LiteLLM auto-falls to next on 429.
AGENT_MODELS = {
    "supervisor": [
        "deepseek/deepseek-reasoner",           # Best reasoning, chain-of-thought
        "groq/llama-3.3-70b-versatile",         # Fast fallback
        "mistral/mistral-large-latest",         # Tertiary fallback
    ],
    "pros": [
        "groq/llama-3.3-70b-versatile",         # 30 req/min, ultra fast
        "cerebras/llama3.1-70b",                # 60 req/min fallback
        "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistral/mistral-small-latest",
    ],
    "cons": [
        "cerebras/llama3.1-70b",                # 60 req/min, fast
        "groq/llama-3.3-70b-versatile",         # fallback
        "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistral/mistral-small-latest",
    ],
    "future": [
    "groq/llama-3.3-70b-versatile",
    "cerebras/llama3.1-70b",
    "mistral/mistral-small-latest",
    "deepseek/deepseek-reasoner",
],
    "summarizer": [
        "groq/llama-3.1-8b-instant",            # Lightweight — summarisation doesn't need 70B
        "mistral/mistral-small-latest",
        "cerebras/llama3.1-70b",
    ],
}

# ── Set API keys for LiteLLM ─────────────────────────────────────────────────
os.environ.setdefault("GROQ_API_KEY",      os.getenv("GROQ_API_KEY", ""))
os.environ.setdefault("CEREBRAS_API_KEY",  os.getenv("CEREBRAS_API_KEY", ""))
os.environ.setdefault("DEEPSEEK_API_KEY",  os.getenv("DEEPSEEK_API_KEY", ""))
os.environ.setdefault("MISTRAL_API_KEY",   os.getenv("MISTRAL_API_KEY", ""))
os.environ.setdefault("TOGETHER_API_KEY",  os.getenv("TOGETHER_API_KEY", ""))


async def call_llm_async(
    agent_type: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """
    Async LLM call with automatic provider failover.
    Never raises on rate limit — just rotates to the next provider.
    """
    models = AGENT_MODELS.get(agent_type, AGENT_MODELS["pros"])
    primary = models[0]
    fallbacks = models[1:] if len(models) > 1 else []

    try:
        response = await acompletion(
            model=primary,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            fallbacks=fallbacks,
            num_retries=3,
            request_timeout=60,
        )
        content = response.choices[0].message.content
        logger.info(f"[{agent_type}] responded via {response.model}")
        return content

    except Exception as e:
        logger.error(f"[{agent_type}] all providers failed: {e}")
        raise RuntimeError(f"All providers exhausted for agent '{agent_type}': {e}")


def call_llm_sync(
    agent_type: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Sync wrapper — use in non-async contexts."""
    models = AGENT_MODELS.get(agent_type, AGENT_MODELS["pros"])
    primary = models[0]
    fallbacks = models[1:] if len(models) > 1 else []

    response = completion(
        model=primary,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        fallbacks=fallbacks,
        num_retries=3,
        request_timeout=60,
    )
    return response.choices[0].message.content
