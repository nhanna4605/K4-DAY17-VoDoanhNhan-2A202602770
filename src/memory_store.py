from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Ước lượng token tất định: ~4 ký tự / token, chuỗi rỗng = 0."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------

PROFILE_HEADER = "# User Profile\n\n"
FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "style": "Style trả lời",
    "interests": "Mối quan tâm",
}
# Khóa đơn trị: correction thay thế giá trị cũ. Khóa đa trị: gộp (giữ thứ tự, không trùng).
MULTI_VALUE_KEYS = {"style", "interests"}
MAX_MULTI_ITEMS = 8
_FACT_LINE = re.compile(r"^- \*\*(\w+)\*\*: (.*)$", re.MULTILINE)


def merge_fact_value(key: str, old: str | None, new: str) -> str:
    """Quy tắc gộp một fact mới vào fact cũ (xử lý conflict/correction)."""

    new = new.strip()
    if old is None or key not in MULTI_VALUE_KEYS:
        return new  # đơn trị: giá trị mới nhất thắng, không giữ song song giá trị cũ
    items = [x.strip() for x in old.split(",") if x.strip()]
    for item in (x.strip() for x in new.split(",")):
        if item and item not in items:
            items.append(item)
    return ", ".join(items[:MAX_MULTI_ITEMS])


@dataclass
class UserProfileStore:
    """Lưu `User.md` của từng user: state/profiles/<user_id>/User.md."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w\-]+", "_", (user_id or "").strip(), flags=re.UNICODE).strip("_")
        return Path(self.root_dir) / (slug or "anonymous") / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if not search_text or search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return {k: v.strip() for k, v in _FACT_LINE.findall(self.read_text(user_id))}

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Ghi một fact theo khóa. Correction sửa đúng dòng cũ (edit_text) thay vì thêm dòng mới."""

        value = (value or "").strip()
        if not value:
            return False
        current = self.facts(user_id)
        if key in current:
            merged = merge_fact_value(key, current[key], value)
            if merged == current[key]:
                return False
            return self.edit_text(
                user_id, f"- **{key}**: {current[key]}", f"- **{key}**: {merged}"
            )
        text = self.read_text(user_id) or PROFILE_HEADER
        if not text.endswith("\n"):
            text += "\n"
        self.write_text(user_id, f"{text}- **{key}**: {merge_fact_value(key, None, value)}\n")
        return True


# ---------------------------------------------------------------------------
# Trích fact từ lời người dùng (kèm confidence threshold)
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 0.6
_HEDGES = ("đùa", "cân nhắc", "có lẽ", "hình như", "giả sử", "ví dụ như")
_PAST_CLAUSE = re.compile(r"(?i)^\s*(?:lúc đầu|ban đầu|trước đây|trước đó|hồi trước)\b")
_CUT_MARKERS = (" chứ không", " dù ")
_ROLE_WORDS = r"engineer|developer|manager|scientist|analyst|designer"

_NAME = re.compile(r"(?i:\b(?:mình|tôi)\s+tên(?:\s+là)?\s+|\btên\s+(?:mình|tôi)\s+là\s+)")
_LOC_SUBJECT = re.compile(
    r"(?i:\b(?:mình|tôi)\s+(?:(?:hiện tại|hiện|đang|vẫn|giờ|làm việc|sống|còn)\s+)*ở\s+)"
)
_LOC_NOW = re.compile(r"(?i:\bhiện(?: tại)?(?:\s+đang)?\s+ở\s+)")
_LOC_PLACE = re.compile(r"(?i:nơi ở(?: hiện tại)?\s+(?:là\s+)?)")
_LOC_CHANGE = re.compile(r"(?i:nơi ở\b[^.]*?\bsang\s+)")
_ROLE = re.compile(
    r"(?i:\b(?:làm|là|chuyển sang|nghề(?: nghiệp)?)\s+)"
    rf"((?:[\w\-]+\s+)?(?i:{_ROLE_WORDS}))\b"
)
_DRINK_IS = re.compile(r"(?i:đồ uống yêu thích(?: của mình)?\s+là\s+)([^.,;]+)")
_DRINK_VERB = re.compile(
    r"(?i:\buống\s+)((?:cà phê|trà|nước|sinh tố|bia)[^.,;]*?)(?=\s+(?:nhưng|như|rồi|và|để|vì)\b|[.,;]|$)"
)
_FOOD_IS = re.compile(r"(?i:món ăn yêu thích(?: của mình)?\s+là\s+)([^.,;]+)")
_PET_RAISE = re.compile(r"(?i:\bnuôi\s+(?:một\s+|1\s+)?(?:bé\s+|con\s+)?)(\S+)(?:\s+tên\s+(\S+))?")
_PET_NAMED = re.compile(r"(?i:\bcon\s+)(corgi|chó|mèo|cún|miu)\s+tên\s+(\S+)")
_INTEREST = re.compile(
    r"(?i:(?<!không )(?<!yêu )(?<!giải )\b(?:thích|quan tâm(?:\s+nhiều)?\s+(?:đến|tới))\s+)(.+)$"
)
_INTEREST_STOP_PREFIX = ("là ", "cách ", "kiểu ", "việc ", "câu ", "những ", "nhiều ", "hơn ")
_INTEREST_CUT = re.compile(r"\s+(?:vì|nhưng|để|hơn|mà|khi)\s+")
_QUESTION_LIKE = re.compile(r"(?i)^\s*(?:nhắc lại|tóm tắt)\b|\bgì\b")


def _proper(rest: str, limit: int = 3) -> str:
    """Lấy tối đa `limit` từ viết hoa liên tiếp ở đầu `rest` (tên riêng/địa danh)."""

    words: list[str] = []
    for token in re.findall(r"[^\s,.;:!?]+", rest):
        if not token[0].isupper() or len(words) >= limit:
            break
        words.append(token)
    return " ".join(words)


def _clean(value: str) -> str:
    return value.strip().strip(".,;:!?\"'")


def _style_items(sentence: str) -> list[str]:
    lowered = sentence.lower()
    if not any(w in lowered for w in ("trả lời", "giải thích", "style", "bullet", "trình bày")):
        return []
    items = []
    if re.search(r"\b(?:ngắn gọn|gọn|ngắn)\b", lowered):
        items.append("ngắn gọn")
    if "3 bullet" in lowered:
        items.append("3 bullet")
    if "ví dụ" in lowered and ("thực tế" in lowered or "thực chiến" in lowered):
        items.append("có ví dụ thực chiến")
    if "trade-off" in lowered:
        items.append("nhấn trade-off")
    return items


def extract_profile_candidates(message: str) -> dict[str, tuple[str, float]]:
    """Trích fact kèm độ tin cậy (0..1). Câu hỏi/câu đùa/giả định bị bỏ hoặc giảm điểm."""

    found: dict[str, tuple[str, float]] = {}

    def put(key: str, value: str, confidence: float, hedged: bool) -> None:
        value = _clean(value)
        if value and value.lower() not in {"gì", "con", "đâu"}:
            found[key] = (value, confidence * (0.5 if hedged else 1.0))

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", message or ""):
        sentence = sentence.strip()
        if not sentence or "?" in sentence:
            continue
        if len(sentence) < 200 and _QUESTION_LIKE.search(sentence):
            continue
        hedged = any(h in sentence.lower() for h in _HEDGES)

        for key in ("style",):
            for item in _style_items(sentence):
                prev = found.get(key, ("", 0.8))[0]
                put(key, merge_fact_value(key, prev or None, item), 0.8, hedged)

        tail = _INTEREST.search(sentence)
        if tail and not re.search(r"(?i)tạm thời|chỉ là", sentence):
            body = _INTEREST_CUT.split(tail.group(1))[0]
            items = []
            for raw in re.split(r",\s*|\s+và\s+|\s+hoặc\s+", body):
                item = _clean(raw)
                if item.startswith("và "):
                    item = item[3:]
                if (
                    item
                    and len(item.split()) <= 3
                    and not re.search(r"(?i)(?:này|đó|ấy|nào)", item)
                    and not item.lower().startswith(_INTEREST_STOP_PREFIX)
                    and item not in items
                ):
                    items.append(item)
            if items:
                prev = found.get("interests", ("", 0.7))[0]
                put("interests", merge_fact_value("interests", prev or None, ", ".join(items)), 0.7, hedged)

        for clause in re.split(r"[,;:]\s*", sentence):
            for marker in _CUT_MARKERS:
                clause = clause.split(marker)[0]
            clause = clause.strip()
            if not clause or _PAST_CLAUSE.match(clause) or re.search(r"(?i)\bkhông còn\b", clause):
                continue
            clause_hedged = hedged or any(h in clause.lower() for h in _HEDGES)

            if m := _NAME.search(clause):
                put("name", _proper(clause[m.end():]), 0.95, clause_hedged)
            for pattern in (_LOC_SUBJECT, _LOC_NOW, _LOC_PLACE):
                if m := pattern.search(clause):
                    place = _proper(clause[m.end():])
                    if place:
                        put("location", place, 0.9, clause_hedged)
                        break
            else:
                if m := _LOC_CHANGE.search(clause):
                    put("location", _proper(clause[m.end():]), 0.9, clause_hedged)
            if m := _ROLE.search(clause):
                put("profession", m.group(1), 0.9, clause_hedged)
            if m := (_DRINK_IS.search(clause) or _DRINK_VERB.search(clause)):
                put("drink", m.group(1), 0.9, clause_hedged)
            if m := _FOOD_IS.search(clause):
                put("food", m.group(1), 0.9, clause_hedged)
            if m := _PET_NAMED.search(clause):
                put("pet", f"{m.group(1)} tên {_clean(m.group(2))}", 0.9, clause_hedged)
            elif m := _PET_RAISE.search(clause):
                animal = _clean(m.group(1))
                put("pet", f"{animal} tên {_clean(m.group(2))}" if m.group(2) else animal, 0.9, clause_hedged)
    return found


def extract_profile_updates(message: str) -> dict[str, str]:
    """Chỉ trả về fact đạt ngưỡng tin cậy `MIN_CONFIDENCE`; lượt chỉ có câu hỏi trả về rỗng."""

    return {
        key: value
        for key, (value, confidence) in extract_profile_candidates(message).items()
        if confidence >= MIN_CONFIDENCE
    }


def facts_from_messages(user_messages: list[str]) -> dict[str, str]:
    """Dựng profile tạm từ các message của MỘT thread (dùng cho short-term memory của baseline)."""

    facts: dict[str, str] = {}
    for message in user_messages:
        for key, value in extract_profile_updates(message).items():
            facts[key] = merge_fact_value(key, facts.get(key), value)
    return facts


# ---------------------------------------------------------------------------
# Trả lời câu hỏi recall từ một bộ fact (dùng chung cho cả hai agent để so sánh công bằng)
# ---------------------------------------------------------------------------

_QUESTION_KEYS = (
    ("name", re.compile(r"(?i)\btên\b")),
    ("location", re.compile(r"(?i)nơi ở|ở đâu|\bở\b")),
    ("profession", re.compile(r"(?i)nghề")),
    ("drink", re.compile(r"(?i)đồ uống")),
    ("food", re.compile(r"(?i)món ăn")),
    ("pet", re.compile(r"(?i)nuôi|thú cưng|con gì")),
    ("style", re.compile(r"(?i)style|trả lời như thế nào|kiểu trả lời|cách trả lời")),
    ("interests", re.compile(r"(?i)quan tâm|sở thích")),
)
_RECALL_REQUEST = re.compile(r"(?i)\?|^\s*(?:nhắc lại|tóm tắt)\b|nhắc lại giúp mình|\bgì\b")


def is_recall_question(message: str) -> bool:
    return len(message) < 300 and bool(_RECALL_REQUEST.search(message or ""))


def answer_from_facts(message: str, facts: dict[str, str]) -> str:
    """Trả lời câu hỏi recall bằng đúng những fact đang có; thiếu thì nói thẳng là chưa biết."""

    keys = [key for key, pattern in _QUESTION_KEYS if pattern.search(message)]
    if not keys:
        keys = [k for k in ("name", "profession", "location", "interests") if k in facts]
    lines = []
    for key in keys:
        value = facts.get(key)
        label = FACT_LABELS[key]
        lines.append(f"- {label}: {value}" if value else f"- {label}: chưa có thông tin")
    if not lines or all(line.endswith("chưa có thông tin") for line in lines):
        return "Mình chưa có thông tin nào về điều này trong cuộc trò chuyện hiện tại."
    return "Theo những gì mình nhớ:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------

SUMMARY_SNIPPET_CHARS = 120
MAX_SUMMARY_LINES = 8


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Tóm tắt heuristic: mỗi message cũ thành một dòng ngắn; chỉ giữ `max_items` dòng cuối."""

    lines = []
    for message in messages[-max_items:]:
        content = " ".join(str(message.get("content", "")).split())
        if len(content) > SUMMARY_SNIPPET_CHARS:
            content = content[:SUMMARY_SNIPPET_CHARS].rstrip() + "…"
        lines.append(f"- {message.get('role', 'user')}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Giữ message gần nhất nguyên văn; khi vượt ngưỡng token thì nén phần cũ thành summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )

    def _tokens(self, thread: dict[str, object]) -> int:
        return estimate_tokens(str(thread["summary"])) + sum(
            estimate_tokens(m["content"]) for m in thread["messages"]
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        # Kiểm tra ngưỡng sau MỖI lần append; chỉ nén được khi còn message cũ ngoài phần giữ lại.
        if self._tokens(thread) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            keep = max(0, self.keep_messages)
            older = thread["messages"][: len(thread["messages"]) - keep]
            thread["messages"] = thread["messages"][len(thread["messages"]) - keep :]
            lines = str(thread["summary"]).splitlines() + summarize_messages(older, len(older)).splitlines()
            thread["summary"] = "\n".join(lines[-MAX_SUMMARY_LINES:])
            thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
