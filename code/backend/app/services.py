from io import BytesIO
import secrets
from threading import RLock
import time

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def decode_image(payload):
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "Ảnh vượt giới hạn 10 MB.")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Chỉ hỗ trợ JPG, PNG, WebP.")
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Ảnh vượt giới hạn 20 triệu pixel.")
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(415, "Nội dung ảnh không hợp lệ.") from exc


class SessionStore:
    def __init__(self, ttl=1800, clock=time.monotonic):
        self.ttl, self.clock = ttl, clock
        self.items = {}
        self.lock = RLock()

    def cleanup(self):
        now = self.clock()
        for key in list(self.items):
            if now - self.items[key]["updated"] > self.ttl:
                del self.items[key]

    def create(self):
        with self.lock:
            self.cleanup()
            if len(self.items) >= 50:
                raise HTTPException(503, "Đã đạt giới hạn phiên, vui lòng thử lại sau.")
            token = secrets.token_urlsafe(32)
            self.items[token] = {"updated": self.clock(), "images": {}, "prompt": "", "last_ids": []}
            return token

    def get(self, token):
        self.cleanup()
        if token not in self.items:
            raise HTTPException(404, "Phiên không tồn tại hoặc đã hết hạn.")
        self.items[token]["updated"] = self.clock()
        return self.items[token]

    def upload(self, token, payloads):
        with self.lock:
            session = self.get(token)
            if len(session["images"]) + len(payloads) > 12:
                raise HTTPException(413, "Mỗi phiên lưu tối đa 12 ảnh; hãy tạo phiên mới.")
            total = sum(len(b) for s in self.items.values() for b in s["images"].values())
            if total + sum(map(len, payloads)) > 128 * 1024 * 1024:
                raise HTTPException(503, "Bộ nhớ ảnh tạm đã đầy.")
            ids = [secrets.token_urlsafe(24) for _ in payloads]
            session["images"].update(zip(ids, payloads))
            return ids

    def resolve(self, token, ids, prompt):
        with self.lock:
            session = self.get(token)
            ids = ids or session["last_ids"]
            prompt = prompt or session["prompt"]
            if not ids or any(i not in session["images"] for i in ids):
                raise HTTPException(404, "Ảnh không thuộc phiên hoặc đã hết hạn.")
            return list(ids), prompt, [session["images"][i] for i in ids]

    def remember(self, token, ids, prompt):
        with self.lock:
            session = self.get(token)
            session["last_ids"], session["prompt"] = list(ids), prompt
