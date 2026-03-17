import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


PROVIDER_DEFAULTS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.0-flash",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
}


def _provider_order() -> List[str]:
    """
    默认顺序：gemini -> deepseek。
    可用 LLM_PROVIDER_ORDER 覆盖，例如："gemini,deepseek"。
    """
    raw = os.getenv("LLM_PROVIDER_ORDER", "gemini,deepseek")
    order = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return [p for p in order if p in PROVIDER_DEFAULTS]


def _build_client(provider: str) -> Optional[Tuple[OpenAI, str, str]]:
    cfg = PROVIDER_DEFAULTS[provider]
    api_key = os.getenv(cfg["api_key_env"], "").strip()
    if not api_key:
        return None

    model = os.getenv(cfg["model_env"], cfg["default_model"]).strip() or cfg["default_model"]
    base_url = cfg["base_url"]

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    return client, model, provider


def chat_completion_with_fallback(
    messages: List[Dict[str, str]],
    model_override: Optional[str] = None,
    response_json: bool = False,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    按 provider 顺序依次尝试调用。
    默认优先 Gemini，用完/失败后回退到 DeepSeek。
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    errors = []
    for provider in _provider_order():
        built = _build_client(provider)
        if not built:
            errors.append(f"{provider}: missing api key")
            continue

        client, default_model, provider_name = built
        model = model_override or default_model

        payload = {
            "model": model,
            "messages": messages,
            **extra_kwargs,
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**payload)
            return resp, provider_name, model
        except Exception as e:
            errors.append(f"{provider_name}: {e}")
            continue

    raise RuntimeError("All providers failed: " + " | ".join(errors))