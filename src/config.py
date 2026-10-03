from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

# Ngưỡng compact: đủ lớn để hội thoại ~10 lượt (standard) không bị nén,
# đủ nhỏ để hội thoại 16 lượt rất dài (stress) bị nén nhiều lần.
DEFAULT_COMPACT_THRESHOLD_TOKENS = 1200
DEFAULT_COMPACT_KEEP_MESSAGES = 4

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}

_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Cấu hình dùng chung cho agent, benchmark và test."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv là tùy chọn khi chạy offline
        return
    load_dotenv(root / ".env", override=False)


def _provider_config(prefix: str, default_provider: str, default_model: str | None) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", default_provider))
    model_name = os.getenv(f"{prefix}_MODEL") or default_model or _DEFAULT_MODEL[provider]
    key_env = _API_KEY_ENV[provider]
    base_url = None
    if provider == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=os.getenv(key_env) if key_env else None,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Nạp .env (nếu có), dựng LabConfig đầy đủ bảy trường và tạo `state/`.

    Biến môi trường (đều tùy chọn — chế độ offline không cần biến nào):
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, JUDGE_PROVIDER, JUDGE_MODEL,
    OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
    CUSTOM_BASE_URL, CUSTOM_API_KEY, OLLAMA_BASE_URL,
    COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, LAB_LIVE (=1 để bật chế độ live).
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM", "openai", None)
    # Judge mặc định dùng cùng provider với model chính, trừ khi JUDGE_* được đặt riêng.
    judge = _provider_config("JUDGE", model.provider, os.getenv("LLM_MODEL") or None)
    if "JUDGE_PROVIDER" not in os.environ:
        judge.api_key, judge.base_url = model.api_key, model.base_url

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(
            os.getenv("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS)
        ),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES)),
        model=model,
        judge_model=judge,
    )
