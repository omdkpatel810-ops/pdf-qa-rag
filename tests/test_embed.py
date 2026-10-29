"""The embedder's own logic (batching, pooling, scaling, errors), with stand-ins for the tokenizer and model.

The real model is tested in test_model.py, which downloads it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdfqa import embed
from pdfqa.embed import QUERY_INSTRUCTION, BgeSmall, EmbeddingError, unit


class FakeEncoding:
    def __init__(self, length: int, padded: int) -> None:
        self.ids = [length] * length + [0] * (padded - length)  # every token id is the text's length
        self.attention_mask = [1] * length + [0] * (padded - length)
        self.type_ids = [0] * padded


class FakeTokenizer:
    """One token per word, padded to the longest text in the batch, like the real one."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.settings: dict[str, object] = {}

    def encode_batch(self, texts):
        self.batches.append(list(texts))
        padded = max(len(t.split()) for t in texts)
        return [FakeEncoding(len(t.split()), padded) for t in texts]

    def enable_truncation(self, length):
        self.settings["truncation"] = length

    def enable_padding(self, **options):
        self.settings["padding"] = options


class FakeSession:
    """Outputs a hidden state whose [CLS] vector is (length, 1, 0, ...): known, and different per text."""

    def __init__(self, *args, **kwargs) -> None:
        self.args, self.kwargs = args, kwargs
        self.padded: list[int] = []

    def run(self, outputs, inputs):
        ids = inputs["input_ids"]
        assert ids.dtype == np.int64 and set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
        self.padded.append(ids.shape[1])
        hidden = np.zeros((ids.shape[0], ids.shape[1], 384), dtype=np.float32)
        hidden[:, 0, 0] = ids[:, 0]
        hidden[:, 0, 1] = 1
        hidden[:, 1:, 2] = 99  # other tokens' vectors must be ignored
        return [hidden]


def loaded(batch_size: int = 2) -> tuple[BgeSmall, FakeTokenizer, FakeSession]:
    model = BgeSmall(batch_size=batch_size)
    model._tokenizer, model._session = FakeTokenizer(), FakeSession()
    return model, model._tokenizer, model._session


def expected(length: int) -> np.ndarray:
    vector = np.zeros(384, dtype=np.float32)
    vector[:2] = length, 1
    return vector / np.linalg.norm(vector)


def test_passages_use_the_cls_vector_scaled_to_length_one():
    model, *_ = loaded()
    texts = ["one two three", "one", "one two three four five", "one two"]

    vectors = model.passages(texts)

    assert vectors.shape == (4, 384) and vectors.dtype == np.float32
    for text, vector in zip(texts, vectors, strict=True):
        assert np.allclose(vector, expected(len(text.split())))


def test_batches_group_similar_lengths_and_report_progress():
    model, tokenizer, session = loaded(batch_size=2)
    seen = []
    model.passages(["a b c d e f", "a", "a b c d e", "a b"], lambda done, total: seen.append((done, total)))
    assert [len(t.split()) for batch in tokenizer.batches for t in batch] == [1, 2, 5, 6]
    assert session.padded == [2, 6]  # the short texts weren't padded to six tokens
    assert seen == [(2, 4), (4, 4)]


def test_questions_get_the_search_instruction():
    model, tokenizer, _ = loaded()
    vector = model.query("How many credit hours?")
    assert tokenizer.batches == [[QUERY_INSTRUCTION + "How many credit hours?"]]
    assert vector.shape == (384,)


def test_nothing_to_embed():
    model = BgeSmall()
    assert model.passages([]).shape == (0, 384)
    assert model._session is None  # the model isn't even loaded


def test_loading_downloads_the_pinned_model(monkeypatch):
    asked = []

    def download(repo, name, revision):
        asked.append((repo, name, revision))
        return f"/cache/{name}"

    tokenizer = FakeTokenizer()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda path: tokenizer)
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)

    model = BgeSmall(threads=2)
    model.passages(["hello there"])
    model.passages(["again"])  # loaded once

    assert asked == [
        (embed.MODEL, "onnx/model.onnx", embed.REVISION),
        (embed.MODEL, "tokenizer.json", embed.REVISION),
    ]
    assert tokenizer.settings == {"truncation": 512, "padding": {"pad_id": 0, "pad_token": "[PAD]"}}
    assert model._session.args[0] == "/cache/onnx/model.onnx"
    assert model._session.args[1].intra_op_num_threads == 2
    assert model.name == "BAAI/bge-small-en-v1.5@5c38ec7"


def test_a_failed_download_says_what_to_do(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("Network is unreachable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)
    with pytest.raises(
        EmbeddingError, match="Couldn't download the BAAI/bge-small-en-v1.5 model .OSError.. It's needed once"
    ):
        BgeSmall().query("anything")


def test_unit_leaves_zero_rows_alone():
    rows = unit(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(rows, [[0.6, 0.8], [0.0, 0.0]]) and rows.dtype == np.float32
