"""Offline frozen CLIP encoder shared by extraction and future inference."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class ClipEncoder:
    def __init__(self, model_path, device="cpu"):
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError("model_path must be an existing local directory")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = device
        self.model_path = path
        self.processor = CLIPProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
        self.model = CLIPModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def encode(self, prompts, images):
        if not prompts or len(prompts) != len(images):
            raise ValueError("expected equal nonempty prompt/image batches")
        if any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("prompts must be nonempty text")
        inputs = self.processor(text=prompts, images=[i.convert("RGB") for i in images],
                                return_tensors="pt", padding=True, truncation=True,
                                max_length=self.model.config.text_config.max_position_embeddings)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(**inputs.to(self.device))
        text = output.text_embeds.float().cpu().numpy()
        image = output.image_embeds.float().cpu().numpy()
        if text.shape != image.shape or text.shape[0] != len(prompts):
            raise ValueError("invalid embedding shape")
        if not np.isfinite(text).all() or not np.isfinite(image).all():
            raise ValueError("nonfinite embedding")
        cosine = np.clip(np.sum(text * image, axis=1), -1, 1)
        return {"text": text, "image": image, "cosine": cosine}
