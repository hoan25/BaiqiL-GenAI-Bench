"""Validate canonical rows before training; no network or dataset downloads."""

from io import BytesIO
from numbers import Integral
from statistics import mean, pstdev
import unicodedata

from PIL import Image, UnidentifiedImageError

MODELS = (
    "DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6",
    "SDXL_2_1", "SDXL_Base", "SDXL_Turbo",
)


def normalize_prompt(prompt):
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    if not normalized:
        raise ValueError("prompt must not be empty")
    return normalized


def validate_ratings(ratings):
    if not isinstance(ratings, (list, tuple)) or len(ratings) != 3:
        raise ValueError("expected exactly three ratings")
    if any(isinstance(r, bool) or not isinstance(r, Integral) or not 1 <= r <= 5
           for r in ratings):
        raise ValueError("ratings must be integers in [1, 5]")
    values = [int(r) for r in ratings]
    return values, mean(values), pstdev(values)


def validate_image(payload):
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("image must contain encoded bytes")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            rgb.load()
            return {"width": rgb.width, "height": rgb.height, "mode": "RGB"}
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"invalid image: {exc}") from exc
