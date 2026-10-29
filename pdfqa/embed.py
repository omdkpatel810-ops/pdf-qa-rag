"""Turn text into vectors that sit close together when they mean similar things.

The model is BAAI's bge-small-en-v1.5: a 33-million-parameter BERT encoder, small enough to run on a laptop
CPU, and among the strongest of its size at retrieval. It runs here with ONNX Runtime, so there's no
PyTorch, no API key and no per-question cost. The steps are the ones the model was trained with:

1. Tokenize with its WordPiece tokenizer, cut to 512 tokens (a 200-word chunk is about 260).
2. Run the encoder. Its first output, at the [CLS] token, stands for the whole text ("CLS pooling").
3. Scale each vector to length 1, so the dot product of two vectors is their cosine similarity.

Questions get the instruction the model was trained to see in front of search queries. Texts go through
one at a time by default: on a 2-core machine that was fastest (10 chunks a second, against 7 in batches of
32), because a batch is padded to its longest text. Larger batches are sorted by length to keep that small.

The model (133 MB) is downloaded from Hugging Face the first time and kept in its usual cache. It's pinned
to one revision, so stored vectors stay comparable with new ones.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pinned: vectors from different revisions don't mix
DIMENSIONS = 384
MAX_TOKENS = 512
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

Progress = Callable[[int, int], None]


class EmbeddingError(Exception):
    """The model can't be loaded or run; the message says why, in plain words."""


class Embedder(Protocol):
    name: str  # stored with every vector, so vectors from different models are never compared
    dimensions: int

    def passages(self, texts: Sequence[str], progress: Progress | None = None) -> np.ndarray:
        """One unit-length float32 row per text."""

    def query(self, text: str) -> np.ndarray:
        """A unit-length float32 vector for a question."""


class BgeSmall:
    """bge-small-en-v1.5 on ONNX Runtime. Loads the model the first time it's needed."""

    name = f"{MODEL}@{REVISION[:7]}"
    dimensions = DIMENSIONS

    def __init__(self, batch_size: int = 1, threads: int | None = None) -> None:
        self.batch_size = batch_size
        self.threads = threads
        self._tokenizer: Any = None
        self._session: Any = None

    def passages(self, texts: Sequence[str], progress: Progress | None = None) -> np.ndarray:
        return self._encode(list(texts), progress)

    def query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_INSTRUCTION + text])[0]

    def _encode(self, texts: list[str], progress: Progress | None = None) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        if not texts:
            return vectors
        tokenizer, session = self._load()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths, less padding
        done = 0
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            encoded = tokenizer.encode_batch([texts[i] for i in batch])
            inputs = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
            }
            hidden = session.run(None, inputs)[0]  # (batch, tokens, 384)
            vectors[batch] = unit(hidden[:, 0, :])  # the [CLS] token's vector
            done += len(batch)
            if progress:
                progress(done, len(texts))
        return vectors

    def _load(self) -> tuple[Any, Any]:
        if self._session is None:
            try:
                import onnxruntime
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
            except ImportError as error:  # pragma: no cover - they're installed with pdfqa
                raise EmbeddingError(f"A package the model needs is missing: {error.name}.") from error
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                model = hf_hub_download(MODEL, "onnx/model.onnx", revision=REVISION)
                vocabulary = hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
            except Exception as error:  # network, disk or Hugging Face errors all mean the same thing here
                raise EmbeddingError(
                    f"Couldn't download the {MODEL} model ({type(error).__name__}). "
                    "It's needed once, then kept on this computer. Check the internet connection and try again."
                ) from error
            tokenizer = Tokenizer.from_file(vocabulary)
            tokenizer.enable_truncation(MAX_TOKENS)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            options = onnxruntime.SessionOptions()
            if self.threads:
                options.intra_op_num_threads = self.threads
            self._tokenizer = tokenizer
            self._session = onnxruntime.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        return self._tokenizer, self._session


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 (rows of zeros stay zeros)."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(lengths == 0, 1, lengths)).astype(np.float32)
