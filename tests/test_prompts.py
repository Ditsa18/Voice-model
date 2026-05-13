from abp_voice.rag.prompts import build_messages, scrub
from abp_voice.rag.store import Retrieved


def _hit(text: str, source: str = "doc.md") -> Retrieved:
    return Retrieved(text=text, metadata={"source": source}, distance=0.1)


def test_build_messages_has_system_and_user():
    msgs = build_messages("What is X?", "en", [_hit("X is a thing.")])
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert "English" in msgs[0]["content"]
    assert "X is a thing." in msgs[1]["content"]


def test_build_messages_clamps_unknown_lang():
    msgs = build_messages("?", "xx", [])
    assert "English" in msgs[0]["content"]


def test_scrub_drops_leaked_system_lines():
    raw = (
        "You are a helpful assistant.\n"
        "Context: blah\n"
        "Q (English): hi\n"
        "Real answer here.\n"
    )
    out = scrub(raw)
    assert "Real answer here." in out
    assert "You are" not in out
    assert "Context:" not in out
    assert "Q (English):" not in out


def test_scrub_passes_clean_answers_through():
    assert scrub("Just an answer.") == "Just an answer."
