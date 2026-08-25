from __future__ import annotations

import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """Singleton BGE-M3 embedding model."""

    _instance = None
    _lock = threading.Lock()

    def __new__(
        cls,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        max_length: int = 8192,
    ):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)

                    instance._initialize(
                        model_name=model_name,
                        device=device,
                        max_length=max_length,
                    )

                    cls._instance = instance

        return cls._instance

    def _initialize(
        self,
        model_name: str,
        device: str | None,
        max_length: int,
    ):
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(
            f">>> Embedding device: {self.device}",
            flush=True,
        )

        if self.device.type == "cuda":
            print(
                f">>> GPU: {torch.cuda.get_device_name(0)}",
                flush=True,
            )

            dtype = torch.float16

        else:
            print(
                ">>> CUDA unavailable. Using CPU.",
                flush=True,
            )

            dtype = torch.float32

        print(
            f">>> Embedding dtype: {dtype}",
            flush=True,
        )

        print(
            f">>> Loading embedding model: "
            f"{model_name}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            ">>> Embedding model loaded",
            flush=True,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> np.ndarray:

        if not texts:
            return np.empty(
                (
                    0,
                    self.model.config.hidden_size,
                ),
                dtype=np.float32,
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        all_embeddings: list[np.ndarray] = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = [
                str(text)
                for text in texts[
                    start:start + batch_size
                ]
            ]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(
                    self.device,
                    non_blocking=True,
                )
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

            token_embeddings = (
                outputs.last_hidden_state
            )

            attention_mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .to(token_embeddings.dtype)
            )

            pooled = (
                token_embeddings * attention_mask
            ).sum(dim=1)

            pooled = pooled / (
                attention_mask.sum(dim=1)
                .clamp(min=1e-9)
            )

            pooled = torch.nn.functional.normalize(
                pooled,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                pooled
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        return np.vstack(all_embeddings)