"""The generation loop, text decoding and the two model backends, with stand-ins for the real models.

The real Qwen3 model is tested in test_model.py, which downloads it.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager

import numpy as np
import pytest

from pdfqa import generate
from pdfqa.generate import (
    ClaudeModel,
    GenerationError,
    LocalModel,
    _penalize,
    chat_prompt,
    detokenize,
    greedy,
    make_writer,
)

VOCAB = 10
LAYERS, KV_HEADS, HEAD_DIM = 2, 1, 3


class CountingModel:
    """A pretend language model: after token t it predicts t + 1, and it checks the cache like the real one.

    Each cached "key" holds the token id it was made for, so tests can check the cache holds every token.
    """

    def __init__(self, scores=None) -> None:
        self.feeds: list[list[int]] = []
        self.scores = scores

    def __call__(self, inputs):
        ids = inputs["input_ids"][0].tolist()
        past = inputs["past_key_values.0.key"]
        seen = past.shape[2]
        assert inputs["attention_mask"].shape == (1, seen + len(ids))
        assert inputs["position_ids"][0].tolist() == list(range(seen, seen + len(ids)))
        self.feeds.append(ids)
        new = np.array(ids, dtype=np.float32).reshape(1, 1, len(ids), 1).repeat(HEAD_DIM, axis=3)
        outputs = {}
        for layer in range(LAYERS):
            for kind in ("key", "value"):
                outputs[f"present.{layer}.{kind}"] = np.concatenate(
                    [inputs[f"past_key_values.{layer}.{kind}"], new], axis=2
                )
        logits = np.zeros((1, len(ids), VOCAB), dtype=np.float32)
        for position, token in enumerate(ids):
            logits[0, position] = self.scores(token) if self.scores else np.eye(VOCAB)[(token + 1) % VOCAB] * 5
        outputs["logits"] = logits
        self.cache = outputs["present.1.value"][0, 0, :, 0].tolist()
        return outputs


def run_greedy(model, prompt, **options):
    settings = {"layers": LAYERS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "stops": [9], "max_tokens": 20}
    return list(greedy(model, prompt, **{**settings, **options}))


def test_reads_the_prompt_in_slices_then_writes_one_token_at_a_time():
    model = CountingModel()
    assert run_greedy(model, [0, 1, 2, 3, 4, 5, 1], step=3) == [2, 3, 4, 5, 6, 7, 8]
    assert model.feeds[:3] == [[0, 1, 2], [3, 4, 5], [1]]  # the prompt, 3 tokens at a time
    assert model.feeds[3:] == [[2], [3], [4], [5], [6], [7], [8]]  # then each new token alone
    assert model.cache == [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8]  # the cache holds every token, in order


def test_stops_after_max_tokens_without_another_pass():
    model = CountingModel()
    assert run_greedy(model, [1], max_tokens=3) == [2, 3, 4]
    assert len(model.feeds) == 3


def test_bad_requests():
    with pytest.raises(ValueError, match="empty"):
        run_greedy(CountingModel(), [])
    with pytest.raises(ValueError, match="max_tokens"):
        run_greedy(CountingModel(), [1], max_tokens=0)


def test_repeated_tokens_score_lower_except_citation_characters():
    # Token 4 always scores 2.0 and token 5 scores 1.9: without a penalty the model would say 4 forever.
    def scores(token):
        row = np.full(VOCAB, -1.0)
        row[4], row[5] = 2.0, 1.9
        return row

    assert run_greedy(CountingModel(scores), [0], max_tokens=4, penalty=1) == [4, 4, 4, 4]
    assert run_greedy(CountingModel(scores), [0], max_tokens=4) == [4, 5, 4, 4]  # 2.0/1.1 < 1.9, then both are cut
    assert run_greedy(CountingModel(scores), [0], max_tokens=4, exempt=[4]) == [4, 4, 4, 4]


def test_penalty_pushes_negative_scores_further_down():
    scores = np.array([2.0, -2.0, 1.0])
    _penalize(scores, [0, 1], 2.0, set())
    assert scores.tolist() == [1.0, -4.0, 1.0]


class ByteTokenizer:
    """Each token is one byte, so a two-byte character arrives in two halves."""

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def test_detokenize_waits_for_whole_characters():
    tokens = list("Café [1]".encode())
    assert list(detokenize(ByteTokenizer(), tokens)) == ["C", "a", "f", "é", " ", "[", "1", "]"]


def test_detokenize_skips_text_that_rewrites_the_past():
    class Rewriting:
        def decode(self, ids):
            return ["ab", "Ab", "Abc"][len(ids) - 1]

    assert list(detokenize(Rewriting(), [1, 2, 3])) == ["ab"]  # "Ab" doesn't extend "ab", so nothing more is safe


def test_chat_prompt_switches_thinking_off():
    text = chat_prompt("Be brief.", "Hi?")
    assert text.startswith("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi?<|im_end|>\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


# The local model, with stand-ins for the tokenizer and ONNX Runtime


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def __init__(self):
        self.encoded = []

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        self.encoded.append(text)
        return FakeEncoding([1, 2])

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def get_vocab(self):
        return {"[": 0, "Ġ[": 1, "]": 2, "7": 3, "hello": 4, "Ġ": 5, "].": 6}


class FakeNode:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Takes everything but position_ids, like some exports, and predicts tokens 3, 4 then the stop token."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.calls = 0
        self.inputs = ["input_ids", "attention_mask"] + [
            f"past_key_values.{i}.{k}" for i in range(LAYERS) for k in ("key", "value")
        ]

    def get_inputs(self):
        return [FakeNode(n) for n in self.inputs]

    def get_outputs(self):
        return [FakeNode("logits")] + [FakeNode(f"present.{i}.{k}") for i in range(LAYERS) for k in ("key", "value")]

    def run(self, names, feed):
        assert names is None and set(feed) == set(self.inputs)  # position_ids was left out
        self.calls += 1
        ids = feed["input_ids"]
        logits = np.zeros((1, ids.shape[1], VOCAB), dtype=np.float32)
        logits[0, -1, [3, 4, 9][min(self.calls - 1, 2)]] = 1
        present = np.zeros((1, KV_HEADS, feed["attention_mask"].shape[1], HEAD_DIM), dtype=np.float32)
        return [logits] + [present] * (2 * LAYERS)


CONFIG = {
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": KV_HEADS,
    "hidden_size": 6,
    "num_attention_heads": 2,
    "eos_token_id": 8,
}


class Downloads:
    """Pretend Hugging Face downloads: files written from `files`, a fake tokenizer and a fake session."""

    def __init__(self, folder):
        self.folder = folder
        self.asked = []
        self.tokenizer = FakeTokenizer()
        self.files = {
            "config.json": json.dumps(CONFIG),
            "generation_config.json": json.dumps({"eos_token_id": [9, 8]}),
            "tokenizer.json": "{}",
            generate.WEIGHTS: "",
        }

    def __call__(self, repo, name, revision):
        self.asked.append((repo, name, revision))
        path = self.folder / name.replace("/", "_")
        path.write_text(self.files[name])
        return str(path)


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    fake = Downloads(tmp_path)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    monkeypatch.setattr("tokenizers.Tokenizer.from_file", lambda path: fake.tokenizer)
    monkeypatch.setattr("onnxruntime.InferenceSession", FakeSession)
    return fake


def test_local_model_downloads_loads_and_writes(downloads):
    model = LocalModel(threads=2)

    assert "".join(model.stream("Be brief.", "Hi?")) == "<3><4>"
    model._load()  # already loaded: nothing is downloaded again

    assert len(downloads.asked) == 4
    assert {name for _, name, _ in downloads.asked} == {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        generate.WEIGHTS,
    }
    assert {(repo, revision) for repo, _, revision in downloads.asked} == {(generate.MODEL, generate.REVISION)}
    assert downloads.tokenizer.encoded == [chat_prompt("Be brief.", "Hi?")]
    assert model._config["stops"] == [9, 8]  # from generation_config.json
    assert model._exempt == {0, 1, 2, 3, 5}  # brackets, digits and spaces; not "hello" or "]."
    assert model._session.args[1].intra_op_num_threads == 2
    assert model._session.args[1].log_severity_level == 3  # ONNX Runtime's warnings stay quiet


def test_local_model_with_a_single_stop_token(downloads):
    downloads.files["generation_config.json"] = "{}"  # then config.json's eos_token_id is used
    model = LocalModel()
    model._load()
    assert model._config["stops"] == [8]


def test_a_failed_download_says_what_to_do(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("Network is unreachable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", offline)
    with pytest.raises(
        GenerationError, match=r"Couldn't download the Qwen3-1.7B answer model \(OSError\). It's 2.1 GB"
    ):
        list(LocalModel().stream("s", "u"))


def test_knows_whether_the_model_is_downloaded(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo, name, revision: "/cache/model.onnx")
    assert LocalModel.is_downloaded()
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo, name, revision: None)
    assert not LocalModel.is_downloaded()


# Claude, with a stand-in for Anthropic's client


class FakeClient:
    def __init__(self, pieces=("Two ", "hours [1]."), error=None):
        self.pieces, self.error, self.calls = pieces, error, []
        self.messages = self

    @contextmanager
    def stream(self, **request):
        self.calls.append(request)
        if self.error:
            raise self.error

        class Reply:
            text_stream = iter(self.pieces)

        yield Reply()


def test_claude_streams_the_reply():
    client = FakeClient()
    model = ClaudeModel("claude-sonnet-5-5", client=client)
    assert "".join(model.stream("Be brief.", "Hi?", 300)) == "Two hours [1]."
    assert client.calls == [
        {
            "model": "claude-sonnet-5-5",
            "max_tokens": 300,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "Hi?"}],
        }
    ]
    assert model.name == "claude-sonnet-5-5 via the Anthropic API"


def test_claude_api_errors_are_explained():
    import anthropic

    error = anthropic.APIConnectionError.__new__(anthropic.APIConnectionError)
    error.message = "Connection error."
    with pytest.raises(GenerationError, match="The Anthropic API said: Connection error."):
        list(ClaudeModel(client=FakeClient(error=error)).stream("s", "u"))


def test_claude_needs_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="needs an API key. Create one at console.anthropic.com"):
        list(ClaudeModel().stream("s", "u"))


def test_claude_makes_a_client_from_the_key(monkeypatch):
    import anthropic

    made = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: made.append(1) or FakeClient(pieces=("ok",)))
    assert "".join(ClaudeModel().stream("s", "u")) == "ok"
    assert made == [1]


def test_claude_needs_its_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(GenerationError, match=r"pip install -e \"\.\[claude\]\""):
        list(ClaudeModel(client=FakeClient()).stream("s", "u"))


def test_choosing_a_model():
    assert isinstance(make_writer("local"), LocalModel)
    assert make_writer("claude").model == generate.CLAUDE_DEFAULT
    assert make_writer("claude-opus-5-5").model == "claude-opus-5-5"
    with pytest.raises(ValueError, match="Unknown model 'gpt-4'"):
        make_writer("gpt-4")
