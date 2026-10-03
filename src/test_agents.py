from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig, normalize_provider


def make_config(tmp_path: Path):
    """Config cô lập cho test: state trong tmp_path, ngưỡng compact nhỏ để nén xảy ra sớm."""

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("dungct") == ""  # chưa có file thì trả rỗng, không ném lỗi

    path = store.write_text("dungct", "# User Profile\n\n- **location**: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert "Đà Nẵng" in store.read_text("dungct")
    assert store.file_size("dungct") == path.stat().st_size > 0

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text("dungct") and "Đà Nẵng" not in store.read_text("dungct")
    assert store.edit_text("dungct", "không tồn tại", "x") is False

    # user_id xấu không được thoát khỏi root_dir
    assert (tmp_path / "profiles") in store.path_for("../../etc/passwd").parents


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    for i in range(8):
        manager.append("t1", "user", f"Tin số {i}: " + "nội dung khá dài " * 12)

    context = manager.context("t1")
    assert manager.compaction_count("t1") > 0
    assert context["summary"]
    assert len(context["messages"]) <= config.compact_keep_messages  # chỉ giữ message gần nhất
    assert manager.compaction_count("thread-khac") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    advanced = AdvancedAgent(make_config(tmp_path / "adv"), force_offline=True)
    baseline = BaselineAgent(make_config(tmp_path / "base"), force_offline=True)
    for agent in (advanced, baseline):
        agent.reply("u1", "thread-1", "Mình tên là DũngCT, mình ở Đà Nẵng.")
        agent.reply("u1", "thread-1", "Mình thường uống cà phê sữa đá.")

    question = "Mình tên gì và đồ uống yêu thích là gì?"
    new_thread_adv = advanced.reply("u1", "thread-2", question)["reply"]
    new_thread_base = baseline.reply("u1", "thread-2", question)["reply"]

    assert "DũngCT" in new_thread_adv and "cà phê sữa đá" in new_thread_adv
    assert "DũngCT" not in new_thread_base and "cà phê sữa đá" not in new_thread_base
    # Baseline vẫn nhớ trong CHÍNH thread đã nói: đó là short-term memory hợp lệ.
    assert "DũngCT" in baseline.reply("u1", "thread-1", "Mình tên gì?")["reply"]
    assert advanced.memory_file_size("u1") > 0 and baseline.compaction_count("thread-1") == 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    advanced = AdvancedAgent(make_config(tmp_path / "adv"), force_offline=True)
    baseline = BaselineAgent(make_config(tmp_path / "base"), force_offline=True)
    for i in range(14):
        text = f"Tin dài số {i}: " + "ngữ cảnh rất dài cần mang theo " * 14
        advanced.reply("u1", "long", text)
        baseline.reply("u1", "long", text)

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")


# --- Bonus: conflict handling + confidence threshold -------------------------------


def test_correction_replaces_old_fact_in_user_md(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u1", "t1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    agent.reply("u1", "t1", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    agent.reply("u1", "t1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    text = agent.profile_store.read_text("u1")
    assert "Huế" in text and "Đà Nẵng" not in text  # không giữ đồng thời fact cũ sai
    assert "MLOps engineer" in text and "backend engineer" not in text
    assert text.count("**location**") == 1

    answer = agent.reply("u1", "t2", "Hiện tại mình làm nghề gì và mình ở đâu?")["reply"]
    assert "MLOps engineer" in answer and "Huế" in answer


def test_noise_and_questions_are_not_written(tmp_path: Path) -> None:
    assert extract_profile_updates("Mình tên gì?") == {}
    assert extract_profile_updates("Hôm nay mình đi họp ở Hà Nội.") == {}
    assert "profession" not in extract_profile_updates(
        "Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa."
    )
    assert normalize_provider("anthorpic") == "anthropic"
    assert load_config(tmp_path).state_dir.exists()
