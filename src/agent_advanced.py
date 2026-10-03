from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    extract_profile_updates,
    is_recall_question,
)
from model_provider import build_chat_model

ACK = "Đã ghi nhận."


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term memory + `User.md` bền vững + compact memory cho thread dài.

    Hai đường độc lập: fact ổn định đi vào User.md (persistent); phần hội thoại đi vào
    CompactMemoryManager (nén khi vượt ngưỡng). Prompt mỗi lượt = User.md + summary + recent.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, message: str) -> list[str]:
        """Bước 1-2: trích fact ổn định rồi ghi vào User.md. Trả về các khóa đã đổi."""

        changed = []
        for key, value in extract_profile_updates(message).items():
            if self.profile_store.upsert_fact(user_id, key, value):
                changed.append(key)
        return changed

    def _finish_turn(self, user_id: str, thread_id: str, message: str, answer: str) -> dict[str, Any]:
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        agent_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        return {"reply": answer, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens}

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed = self._remember(user_id, message)  # 1-2. trích fact -> ghi User.md
        self.compact_memory.append(thread_id, "user", message)  # 3. short-term + tự compact
        if is_recall_question(message):
            answer = self._offline_response(user_id, thread_id, message)  # 5
        else:
            answer = ACK + (f" Đã lưu vào User.md: {', '.join(changed)}." if changed else "")
        return self._finish_turn(user_id, thread_id, message, answer)  # 4 + 6

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Ngữ cảnh mang vào một lượt = User.md + summary + các message gần nhất còn nguyên văn."""

        context = self.compact_memory.context(thread_id)
        recent = sum(estimate_tokens(m["content"]) for m in context["messages"])
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(context["summary"]))
            + recent
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Trả lời tất định bằng fact đã lưu trong User.md (đúng cả khi sang thread mới)."""

        return answer_from_facts(message, self.profile_store.facts(user_id))

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        context = self.compact_memory.context(thread_id)
        system = (
            "Bạn là trợ lý có bộ nhớ. Hồ sơ người dùng (User.md):\n"
            f"{self.profile_store.read_text(user_id) or '(chưa có)'}\n"
            f"Tóm tắt hội thoại cũ:\n{context['summary'] or '(chưa có)'}"
        )
        messages = [{"role": "system", "content": system}, *context["messages"]]
        answer = str(self.langchain_agent.invoke(messages).content)
        return self._finish_turn(user_id, thread_id, message, answer)

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
