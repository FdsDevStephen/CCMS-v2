from __future__ import annotations

import gc
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMBEDDING_DIM = 1024  # BGE-M3


def _free_memory_gb() -> float | None:
    """
    Free commit memory in GB, or None if unknown.

    On Windows this is the commit limit left (what "The paging file is
    too small" errors run out of); psutil would overcount it by adding
    unused page-file space.
    """
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)

        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPageFile / 2**30
    except Exception:
        pass

    try:
        import psutil

        return (
            psutil.virtual_memory().available
            + psutil.swap_memory().free
        ) / 2**30
    except Exception:
        pass

    return None


def _loaded_llm_gb() -> float:
    """
    Size in GB of the configured Ollama model if it is already loaded.

    That memory is already counted as used, so "auto" should not demand
    free room for it a second time.
    """
    try:
        import json
        import urllib.request

        from config.settings import LLM_BASE_URL, MODEL_NAME

        with urllib.request.urlopen(
            f"{LLM_BASE_URL}/api/ps", timeout=2
        ) as response:
            models = json.load(response).get("models", [])

        return sum(
            m.get("size", 0) for m in models
            if m.get("name") == MODEL_NAME
        ) / 2**30
    except Exception:
        return 0.0


def _cpu_dtype() -> torch.dtype:
    from config.settings import (
        EMBEDDING_CPU_DTYPE,
        EMBEDDING_FLOAT32_MIN_FREE_GB,
    )

    choice = EMBEDDING_CPU_DTYPE.strip().lower()

    if choice != "auto":
        return getattr(torch, choice)

    free = _free_memory_gb()

    if free is None:
        return torch.float32

    from config.settings import LLM_PROVIDER

    if LLM_PROVIDER.lower() == "ollama":
        needed = EMBEDDING_FLOAT32_MIN_FREE_GB - _loaded_llm_gb()
    else:
        # Remote LLM: only BGE-M3 itself (~4 GB in float32) must fit.
        needed = 4.5

    if free >= needed:
        return torch.float32

    print(
        f">>> Only {free:.1f} GB memory free (float32 needs "
        f"{needed:.1f} GB): loading BGE-M3 in bfloat16 (smaller, "
        f"~5x slower embeddings) so Ollama still fits. Close other "
        f"programs and restart for the fast mode.",
        flush=True,
    )
    return torch.bfloat16


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
        self.remote_url = None

        from config.settings import EMBEDDING_PROVIDER, EMBEDDING_URL

        if EMBEDDING_PROVIDER.lower() == "remote":
            # Hosted BGE-M3: no model in local memory. The tokenizer is
            # still needed locally by the chunker (it is small).
            self.remote_url = EMBEDDING_URL
            self.model = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )

            print(
                f">>> Embeddings: remote BGE-M3 at {EMBEDDING_URL}",
                flush=True,
            )
            return

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

            dtype = _cpu_dtype()

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

        # Free the temporary state dict left over from loading
        # pytorch_model.bin, so it doesn't sit in RAM next to Ollama.
        gc.collect()

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
                    EMBEDDING_DIM,
                ),
                dtype=np.float32,
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if self.remote_url:
            return self._encode_remote(
                texts,
                batch_size,
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
    def _encode_remote(
        self,
        texts: list[str],
        batch_size: int,
    ) -> np.ndarray:
        """POST {"texts": [...]} -> {"embeddings": [[...], ...]}."""
        import time

        import requests

        from config.settings import MAX_RETRIES, REQUEST_TIMEOUT

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = [str(text) for text in texts[start:start + batch_size]]

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = requests.post(
                        self.remote_url,
                        json={"texts": batch},
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    embeddings = response.json()["embeddings"]
                    break
                except Exception as e:
                    print(
                        f"[Embeddings] Attempt {attempt}/{MAX_RETRIES} "
                        f"failed: {e}",
                        flush=True,
                    )

                    if attempt == MAX_RETRIES:
                        raise RuntimeError(
                            f"Remote embedding service at "
                            f"{self.remote_url} failed after "
                            f"{MAX_RETRIES} attempts."
                        ) from e

                    time.sleep(2)

            all_embeddings.append(
                np.asarray(embeddings, dtype=np.float32)
            )

        return np.vstack(all_embeddings)
