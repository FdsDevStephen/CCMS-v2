from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """BGE-M3 embedding model with mean pooling and L2 normalization."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        max_length: int = 8192,
    ):
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = [str(text) for text in texts[start:start + batch_size]]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()

            pooled = (token_embeddings * attention_mask).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1).clamp(min=1e-9)

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)
