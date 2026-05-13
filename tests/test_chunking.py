from abp_voice.rag.chunking import chunk_text, hash_id, normalize_whitespace


def test_normalize_collapses_runs():
    assert normalize_whitespace("a  \t  b\n\n\n\nc") == "a b\n\nc"


def test_empty_text_yields_no_chunks():
    assert chunk_text("", size=200, overlap=20) == []


def test_chunks_respect_size():
    text = "abcdefghij" * 50  # 500 chars, no natural breaks
    chunks = chunk_text(text, size=120, overlap=20)
    assert all(len(c) <= 120 for c in chunks)
    assert len(chunks) >= 4


def test_prefers_sentence_break():
    text = (
        "First sentence here that is quite long. " * 5
        + "Second paragraph starts here and continues on."
    )
    chunks = chunk_text(text, size=150, overlap=10)
    # Most chunk boundaries should land after ". " not mid-word.
    assert any(c.endswith(".") for c in chunks)


def test_hash_id_stable_and_short():
    a = hash_id("hello world", "doc.txt", 0)
    b = hash_id("hello world", "doc.txt", 0)
    assert a == b
    assert len(a) == 24
