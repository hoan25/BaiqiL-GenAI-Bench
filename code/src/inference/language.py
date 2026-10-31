"""Cheap prompt-language check used to decide whether the English-only ITM gate may run.

No external model: Vietnamese is recognised by its diacritics / ``đ`` or by a few
diacritic-free function words; any other letter outside the Latin blocks counts
as a non-Latin script. Everything else is treated as Latin text (English-like).
"""

import re
import unicodedata

# Vietnamese-only letters (both cases) including the horn/hook/tone combinations.
_VIETNAMESE_LETTERS = re.compile(
    "[ăâêôơưđĂÂÊÔƠƯĐ"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ]")

# Short list of frequent words that stay unambiguous without diacritics
# (text typed on a keyboard without Vietnamese input). Two hits are required
# so a stray English "con" or "trong" does not trigger.
_VIETNAMESE_WORDS = {"mot", "nhung", "khong", "nguoi", "dang", "tren", "duoi", "buc", "hinh", "mau",
                     "con", "cai", "chiec", "ngoi", "cay", "nha", "meo", "cho", "voi", "cua",
                     "trong", "ngoai", "truoc", "sau", "canh", "troi", "nuoc", "duong", "xe", "ben"}


def _is_non_latin_letter(char):
    if not char.isalpha():
        return False
    code = ord(char)
    # Basic Latin, Latin-1, Latin Extended-A/B, IPA, Latin Extended Additional (Vietnamese).
    if code < 0x0250 or 0x1E00 <= code <= 0x1EFF:
        return False
    return unicodedata.category(char).startswith("L")


def prompt_language(text):
    """Return ``"vi"``, ``"non_latin"`` or ``"latin"`` for the given prompt."""
    if not isinstance(text, str):
        raise TypeError("prompt must be a string")
    text = unicodedata.normalize("NFC", text)
    if _VIETNAMESE_LETTERS.search(text):
        return "vi"
    if any(_is_non_latin_letter(c) for c in text):
        return "non_latin"
    words = re.findall(r"[a-z]+", text.casefold())
    if sum(w in _VIETNAMESE_WORDS for w in words) >= 2:
        return "vi"
    return "latin"


def gate_supported(text):
    """True when the English-only matcher may be trusted on this prompt."""
    return prompt_language(text) == "latin"
