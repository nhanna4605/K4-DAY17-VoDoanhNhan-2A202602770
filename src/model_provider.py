from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Cấu hình provider dùng chung cho cả hai agent và judge model."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Chuẩn hóa alias (ví dụ `anthorpic` -> `anthropic`); provider lạ thì báo lỗi rõ ràng."""

    key = (value or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Provider không được hỗ trợ: {value!r}. Chọn một trong: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def _require(package: str, provider: str):
    try:
        return __import__(package, fromlist=["*"])
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError(
            f"Provider '{provider}' cần package `{package.replace('_', '-')}`. "
            f"Cài bằng: pip install {package.replace('_', '-')}"
        ) from exc


def build_chat_model(config: ProviderConfig):
    """Dựng chat model thật theo provider. Import lười để chế độ offline không cần SDK nào."""

    provider = normalize_provider(config.provider)
    kwargs = {"model": config.model_name, "temperature": config.temperature}

    if provider in ("openai", "custom"):
        module = _require("langchain_openai", provider)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("Provider 'custom' cần base_url (CUSTOM_BASE_URL).")
            kwargs["base_url"] = config.base_url
        return module.ChatOpenAI(**kwargs)

    if provider == "gemini":
        module = _require("langchain_google_genai", provider)
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return module.ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        module = _require("langchain_anthropic", provider)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return module.ChatAnthropic(**kwargs)

    if provider == "ollama":
        module = _require("langchain_ollama", provider)
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return module.ChatOllama(**kwargs)

    module = _require("langchain_openrouter", provider)
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return module.ChatOpenRouter(**kwargs)
