import pytest

from src.inference.language import gate_supported, prompt_language


@pytest.mark.parametrize("text", [
    "a red bicycle leaning against a stone wall",
    "A photo of two cats on a sofa.",
    "Un chat noir sur la table",              # French stays Latin: BLIP is unreliable but not blocked
    "The con artist walked through the park",  # single ambiguous word must not trigger
    "x",
])
def test_latin_prompts_keep_the_gate(text):
    assert prompt_language(text) == "latin"
    assert gate_supported(text)


@pytest.mark.parametrize("text", [
    "một con mèo đang ngồi trên ghế",
    "Bức ảnh có cây cầu đỏ",
    "ĐƯỜNG PHỐ VỀ ĐÊM",
    "mot con meo dang ngoi tren ghe",        # typed without diacritics
    "hinh anh mau do",
])
def test_vietnamese_detected(text):
    assert prompt_language(text) == "vi"
    assert not gate_supported(text)


@pytest.mark.parametrize("text", ["赤い自転車", "кошка на диване", "قطة على الأريكة", "a cat 猫"])
def test_non_latin_scripts_detected(text):
    assert prompt_language(text) == "non_latin"
    assert not gate_supported(text)


def test_decomposed_unicode_is_normalised():
    assert prompt_language("mèo") == "vi"


def test_rejects_non_string():
    with pytest.raises(TypeError):
        prompt_language(None)
