# llm.py — provider/model selection shared by the proposal agents.
#
# All three providers expose an OpenAI-compatible Chat Completions endpoint, so a
# single `openai.OpenAI` client works for each — only base_url, model, and key change.
# The UI writes the user's choice into the LLM_PROVIDER / LLM_MODEL / LLM_API_KEY
# environment variables; the agents read them here at call time.
import os
from openai import OpenAI

# provider -> { base_url, key_prefix (UI hint only), models (first = default) }
PROVIDERS = {
    "Anthropic": {
        # Anthropic's OpenAI-SDK compatibility endpoint; models are bare IDs.
        "base_url": "https://api.anthropic.com/v1/",
        "key_prefix": "sk-ant-...",
        "models": [
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-opus-4-7",
            "claude-fable-5",
        ],
    },
    "Grok (xAI)": {
        "base_url": "https://api.x.ai/v1",
        "key_prefix": "xai-...",
        "models": [
            "grok-4",
            "grok-3",
            "grok-3-mini",
        ],
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "key_prefix": "sk-...",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
        ],
    },
}

DEFAULT_PROVIDER = "Anthropic"


def get_llm() -> tuple[OpenAI, str]:
    """Build an OpenAI-compatible client + model name from the env-configured provider."""
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    cfg = PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip() or cfg["models"][0]
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key)
    return client, model
