from abp_voice.languages import (
    EDGE_VOICES,
    LANG_NAMES,
    SCRIPT_HINT,
    SUPPORTED_LANGS,
    is_exit_phrase,
    normalize_lang,
)


def test_supported_langs_cover_three():
    assert set(SUPPORTED_LANGS) == {"en", "hi", "bn"}


def test_each_supported_lang_has_metadata():
    for code in SUPPORTED_LANGS:
        assert code in LANG_NAMES
        assert code in EDGE_VOICES
        assert code in SCRIPT_HINT


def test_normalize_lang_clamps_unknown():
    assert normalize_lang("fr") == "en"
    assert normalize_lang(None) == "en"
    assert normalize_lang("") == "en"
    assert normalize_lang("bn") == "bn"


def test_exit_phrases_match_each_language():
    assert is_exit_phrase("please exit")
    assert is_exit_phrase("बंद करो अब")
    assert is_exit_phrase("বিদায় বন্ধু")
    assert not is_exit_phrase("hello there")
    assert not is_exit_phrase("")
