from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    answer_from_facts,
    estimate_tokens,
    facts_from_messages,
    is_recall_question,
)
from model_provider import build_chat_model

ACK = "Đã ghi nhận."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: chỉ có short-term memory trong cùng một thread.

    `self.sessions` được khóa theo `thread_id` (KHÔNG phải user_id), nên sang thread mới
    agent này quên sạch mọi fact. Không có User.md, không compact.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _record(self, thread_id: str, message: str, answer: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        # Baseline mang TOÀN BỘ lịch sử của thread vào mọi lượt: cộng dồn ngay trong từng lượt.
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        agent_tokens = estimate_tokens(answer)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += agent_tokens
        session.messages.append({"role": "assistant", "content": answer})
        return {"reply": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        if is_recall_question(message):
            # Chỉ dùng những gì đã nói TRONG thread này (short-term memory).
            user_messages = [m["content"] for m in session.messages if m["role"] == "user"]
            answer = answer_from_facts(message, facts_from_messages(user_messages))
        else:
            answer = ACK
        return self._record(thread_id, message, answer)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        history = [*session.messages, {"role": "user", "content": message}]
        answer = str(self.langchain_agent.invoke(history).content)
        return self._record(thread_id, message, answer)

    def _maybe_build_langchain_agent(self):
        """Chỉ dựng model thật khi LAB_LIVE=1 và có đủ SDK/API key; ngược lại rơi về offline."""

        if os.getenv("LAB_LIVE") != "1":
            return None
        cfg = self.config.model
        if cfg.provider != "ollama" and not cfg.api_key:
            return None
        try:
            return build_chat_model(cfg)
        except (RuntimeError, ValueError):
            return None
