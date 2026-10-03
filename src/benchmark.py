from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _hits(answer: str, expected: list[str]) -> int:
    folded = answer.casefold()
    return sum(1 for item in expected if item.casefold() in folded)


def recall_points(answer: str, expected: list[str]) -> float:
    """0 nếu không fact nào xuất hiện, 1 nếu đủ hết, 0.5 nếu đúng một phần."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == 0:
        return 0.0
    return 1.0 if hits == len(expected) else 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Điểm chất lượng offline 0..1, cùng một công thức cho mọi agent.

    70% độ phủ fact cần có + 20% độ gọn (câu trả lời ngắn được ưu tiên)
    + 10% không thừa nhận "chưa có thông tin".
    """

    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    length = len(answer.strip())
    concise = 1.0 if 0 < length <= 400 else 0.5 if 0 < length <= 800 else 0.0
    confident = 0.0 if "chưa có thông tin" in answer else 1.0
    return round(0.7 * coverage + 0.2 * concise + 0.1 * confident, 4)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Chạy một agent trên toàn bộ hội thoại; recall luôn được hỏi ở thread MỚI."""

    agent_tokens = prompt_tokens = compactions = 0
    recalls: list[float] = []
    qualities: list[float] = []
    sizes_before: dict[str, int] = {}
    users: list[str] = []

    for conv in conversations:
        user_id = conv["user_id"]
        if user_id not in sizes_before:
            users.append(user_id)
            sizes_before[user_id] = _memory_size(agent, user_id)

        chat_thread = f"{agent_name}-{conv['id']}-chat"
        for turn in conv["turns"]:
            agent.reply(user_id, chat_thread, turn)
        threads = [chat_thread]

        for index, item in enumerate(conv["recall_questions"], start=1):
            recall_thread = f"{agent_name}-{conv['id']}-recall-{index}"
            answer = agent.reply(user_id, recall_thread, item["question"])["reply"]
            recalls.append(recall_points(answer, item["expected_contains"]))
            qualities.append(heuristic_quality(answer, item["expected_contains"]))
            threads.append(recall_thread)

        for thread in threads:
            agent_tokens += agent.token_usage(thread)
            prompt_tokens += agent.prompt_token_usage(thread)
            compactions += agent.compaction_count(thread)

    growth = sum(_memory_size(agent, user) - sizes_before[user] for user in users)
    count = max(1, len(recalls))
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recalls) / count, 3),
        response_quality=round(sum(qualities) / count, 3),
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def _memory_size(agent, user_id: str) -> int:
    size = getattr(agent, "memory_file_size", None)
    return size(user_id) if size else 0


HEADERS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        (
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        )
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=HEADERS, tablefmt="github")
    except ImportError:  # tabulate là tùy chọn: tự dựng bảng markdown
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def run_suite(title: str, dataset: Path, config) -> str:
    """Chạy Baseline và Advanced trên CÙNG dữ liệu, mỗi agent với state sạch riêng."""

    conversations = load_conversations(dataset)
    rows = []
    for name, factory in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        state_dir = config.state_dir / "benchmark" / dataset.stem / name.lower()
        shutil.rmtree(state_dir, ignore_errors=True)
        agent = factory(replace(config, state_dir=state_dir), force_offline=True)
        rows.append(run_agent_benchmark(name, agent, conversations, config))
    return f"## {title}\n\n{format_rows(rows)}\n"


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    print(run_suite("Standard Benchmark", config.data_dir / "conversations.json", config))
    print(run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config))


if __name__ == "__main__":
    main()
