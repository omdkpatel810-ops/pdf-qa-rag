"""Write text with a language model: a small one on your own computer (the default), or Claude.

The local model is Qwen3-1.7B (Apache 2.0), 4-bit, on ONNX Runtime: the same runtime as the embedding model,
so there's still no PyTorch, no API key and no per-question cost. The 4-bit version beat the 8-bit one on
five test questions, 4 right to 3, and how they failed matters more: 4-bit's miss was a cautious "the
documents don't say", while 8-bit twice made up a wrong answer from the wrong passages. ONNX Runtime runs one forward pass; the
loop that turns passes into text is here:

1. **Prefill.** The prompt goes through the model in slices of 256 tokens. Each pass returns the attention
   keys and values for its tokens (the "KV cache"), which the next pass reads instead of recomputing them.
   Slicing keeps memory flat: every pass also returns a score for every vocabulary entry at every position
   (151,936 floats each), and for a 1,200-token prompt in one pass that alone would be 730 MB.
2. **Decode.** One token at a time: the model sees only the newest token plus the cache, and the
   highest-scoring next token is taken (greedy, so the same question gets the same answer, which keeps the
   accuracy test in milestone 4 repeatable).
3. **Repetition penalty.** Tokens already written score a little lower (÷1.1), the usual guard against
   greedy loops, except brackets and digits, so citations like [2] stay as likely as the model wants.
4. **Stop** at the end-of-turn token or after `max_tokens`.

Qwen3 can "think" before answering; the prompt opens with an empty thinking block, which is how Qwen
switches that off, because short answers from given sources don't need it and it would triple the wait.

Claude (`--model claude`) uses Anthropic's API through its official SDK, installed with
`pip install -e ".[claude]"`, and the key you set in ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

import numpy as np

MODEL = "onnx-community/Qwen3-1.7B-ONNX"
REVISION = "3231b522c6e703b2f9b3cd8ea27426a1289a02f2"  # pinned, so answers don't change under milestone 4's test
WEIGHTS = "onnx/model_q4.onnx"  # 2.1 GB
DOWNLOAD_SIZE = "2.1 GB"
CLAUDE_DEFAULT = "claude-haiku-4-5"
PREFILL_STEP = 256
PENALTY = 1.1
_CITATION_CHARACTERS = set(" []0123456789")

Run = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


class GenerationError(Exception):
    """The model can't be loaded or run; the message says why, in plain words."""


class Writer(Protocol):
    name: str  # shown with every answer, e.g. "Qwen3-1.7B on this computer"

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        """The reply to one message, piece by piece as it's written."""


def chat_prompt(system: str, user: str) -> str:
    """Qwen's chat format (ChatML), with an empty thinking block so it answers straight away."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def greedy(
    run: Run,
    prompt: list[int],
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    stops: Iterable[int],
    max_tokens: int,
    dtype: Any = np.float32,
    step: int = PREFILL_STEP,
    penalty: float = PENALTY,
    exempt: Iterable[int] = (),
) -> Iterator[int]:
    """Token ids written after `prompt`, one at a time. `run` is one forward pass of the model."""
    if not prompt:
        raise ValueError("The prompt is empty.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1.")
    stops, exempt = set(stops), set(exempt)
    names = [(f"past_key_values.{i}.{kind}", f"present.{i}.{kind}") for i in range(layers) for kind in ("key", "value")]
    cache = {past: np.zeros((1, kv_heads, 0, head_dim), dtype=dtype) for past, _ in names}
    pending, position, written = list(prompt), 0, []
    while True:
        feed, pending = pending[:step], pending[step:]
        outputs = run(
            {
                "input_ids": np.array([feed], dtype=np.int64),
                "attention_mask": np.ones((1, position + len(feed)), dtype=np.int64),
                "position_ids": np.arange(position, position + len(feed), dtype=np.int64)[None, :],
                **cache,
            }
        )
        position += len(feed)
        cache = {past: outputs[present] for past, present in names}
        if pending:  # still reading the prompt
            continue
        scores = outputs["logits"][0, -1].astype(np.float64)
        _penalize(scores, written, penalty, exempt)
        token = int(np.argmax(scores))
        if token in stops:
            return
        written.append(token)
        yield token
        if len(written) >= max_tokens:
            return
        pending = [token]


def _penalize(scores: np.ndarray, written: list[int], penalty: float, exempt: set[int]) -> None:
    """Lower the scores of tokens already written (CTRL's rule: divide if positive, multiply if negative)."""
    repeated = np.fromiter(set(written) - exempt, dtype=np.int64)
    if penalty != 1 and len(repeated):
        values = scores[repeated]
        scores[repeated] = np.where(values > 0, values / penalty, values * penalty)


def detokenize(tokenizer: Any, tokens: Iterable[int]) -> Iterator[str]:
    """Text for a stream of token ids, as it grows. Holds back half-finished multi-byte characters."""
    written: list[int] = []
    shown = ""
    for token in tokens:
        written.append(token)
        text = tokenizer.decode(written)
        if text.endswith("�") or not text.startswith(shown):
            continue  # wait for the rest of the character
        if len(text) > len(shown):
            yield text[len(shown) :]
            shown = text


class LocalModel:
    """Qwen3-1.7B on ONNX Runtime. Downloads (2.1 GB, once) and loads the first time it's needed."""

    name = "Qwen3-1.7B on this computer"

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads
        self._tokenizer: Any = None
        self._session: Any = None
        self._config: dict[str, Any] = {}
        self._exempt: set[int] = set()

    @staticmethod
    def is_downloaded() -> bool:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(MODEL, WEIGHTS, revision=REVISION), str)

    def stream(self, system: str, user: str, max_tokens: int = 400) -> Iterator[str]:
        tokenizer, session = self._load()
        prompt = tokenizer.encode(chat_prompt(system, user), add_special_tokens=False).ids
        config = self._config
        tokens = greedy(
            self._run,
            prompt,
            layers=config["num_hidden_layers"],
            kv_heads=config["num_key_value_heads"],
            head_dim=config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"],
            stops=config["stops"],
            max_tokens=max_tokens,
            exempt=self._exempt,
        )
        yield from detokenize(tokenizer, tokens)

    def _run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {name: value for name, value in inputs.items() if name in self._config["inputs"]}
        return dict(zip(self._config["outputs"], self._session.run(None, feed), strict=True))

    def _load(self) -> tuple[Any, Any]:
        if self._session is None:
            import json

            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                config = json.loads(open(hf_hub_download(MODEL, "config.json", revision=REVISION)).read())
                generation = json.loads(
                    open(hf_hub_download(MODEL, "generation_config.json", revision=REVISION)).read()
                )
                vocabulary = hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
                weights = hf_hub_download(MODEL, WEIGHTS, revision=REVISION)
            except Exception as error:  # network, disk or Hugging Face errors all mean the same thing here
                raise GenerationError(
                    f"Couldn't download the Qwen3-1.7B answer model ({type(error).__name__}). It's {DOWNLOAD_SIZE}, "
                    "needed once, then kept on this computer. Check the internet connection and try again, "
                    "or answer with Claude instead: --model claude (needs your own API key)."
                ) from error
            stops = generation.get("eos_token_id", config["eos_token_id"])
            config["stops"] = stops if isinstance(stops, list) else [stops]
            tokenizer = Tokenizer.from_file(vocabulary)
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3  # errors only: loading prints harmless optimizer warnings otherwise
            if self.threads:
                options.intra_op_num_threads = self.threads
            session = onnxruntime.InferenceSession(weights, options, providers=["CPUExecutionProvider"])
            config["inputs"] = {i.name for i in session.get_inputs()}
            config["outputs"] = [o.name for o in session.get_outputs()]
            self._exempt = {
                token_id
                for piece, token_id in tokenizer.get_vocab().items()
                if set(piece.replace("Ġ", " ")) <= _CITATION_CHARACTERS
            }
            self._config, self._tokenizer, self._session = config, tokenizer, session
        return self._tokenizer, self._session


class ClaudeModel:
    """Claude through Anthropic's API. Needs `pip install -e ".[claude]"` and your key in ANTHROPIC_API_KEY."""

    def __init__(self, model: str = CLAUDE_DEFAULT, client: Any = None) -> None:
        self.model = model
        self.name = f"{model} via the Anthropic API"
        self._client = client

    def stream(self, system: str, user: str, max_tokens: int = 1024) -> Iterator[str]:
        try:
            import anthropic
        except ImportError as error:
            raise GenerationError(
                'Answering with Claude needs Anthropic\'s package. Install it with: pip install -e ".[claude]"'
            ) from error
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise GenerationError(
                    "Answering with Claude needs an API key. Create one at console.anthropic.com, then set it "
                    "in your terminal: export ANTHROPIC_API_KEY=... (never put it in a file you commit)."
                )
            self._client = anthropic.Anthropic()
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as reply:
                yield from reply.text_stream
        except anthropic.APIError as error:
            raise GenerationError(f"The Anthropic API said: {error.message}") from error


def make_writer(name: str) -> Writer:
    """ "local" for Qwen3 on this computer, "claude" for Claude's default, or a Claude model id."""
    if name == "local":
        return LocalModel()
    if name == "claude":
        return ClaudeModel()
    if name.startswith("claude-"):
        return ClaudeModel(name)
    raise ValueError(f"Unknown model {name!r}. Use local, claude, or a Claude model id like {CLAUDE_DEFAULT}.")
