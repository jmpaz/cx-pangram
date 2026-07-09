"""EditLens inference: load an open-pangram checkpoint and score text for AI-edit extent.

Decoding mirrors pangramlabs/EditLens (scripts/inference.py):

    score = (softmax(logits) · arange(n_buckets)) / (n_buckets - 1)

i.e. the expected bucket index normalized to [0, 1]; 0 = fully human, 1 = fully
AI-generated, and intermediate values quantify the degree of AI editing applied to a
human draft. Long inputs are windowed into chunks (EditLens was trained on 75-799 word
texts at <=512 tokens); chunk scores are length-weighted into a document aggregate while
the per-chunk vector is retained for annotation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import cast

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from . import ModelAccessError
from . import bands as _bands
from .bands import Band
from .preprocess import clean_text

MODELS: dict[str, tuple[str, str]] = {
    "roberta": ("pangram/editlens_roberta-large", "FacebookAI/roberta-large"),
    "llama": ("pangram/editlens_Llama-3.2-3B", "unsloth/Llama-3.2-3B"),
}

MAX_LENGTH = 512
LLAMA_MAX_LENGTH = 1024
MIN_WORDS = 50
CHUNK_WORDS = 350

DEFAULT_BATCH = 8
MAX_BATCH = 64
VRAM_BUDGET = 0.5
ACT_SAFETY = 24  # rough per-window activation multiplier


def select_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def default_model(device: str) -> str:
    return "llama" if device.startswith(("cuda", "mps")) else "roberta"


def _unquantized_dtype(device: str):
    if device.startswith(("cuda", "mps")):
        return torch.bfloat16
    return torch.float32


def _bnb_usable(device: str) -> bool:
    if not device.startswith("cuda") or torch.version.hip is not None:
        return False
    try:
        import bitsandbytes  # noqa: F401  # ty: ignore[unresolved-import]

        return True
    except Exception:
        return False


def _is_oom(exc: RuntimeError) -> bool:
    """CUDA raises a dedicated type; MPS and CPU raise plain RuntimeErrors whose
    message is the only signal."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _empty_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device.startswith("mps"):
        torch.mps.empty_cache()


def _resolve_quantize(device: str, want: bool | None) -> bool:
    if want is None:
        return _bnb_usable(device)
    if want:
        if not _bnb_usable(device):
            raise RuntimeError(
                f"quantize=True requested but 4-bit bitsandbytes is unavailable on "
                f"{device}; needs a real CUDA device with cx-pangram[cuda] installed"
            )
        return True
    return False


class NormedLinear(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(
            hidden_size, num_labels, bias=False, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.linear(self.norm(x))


@contextmanager
def _hf_access(repo: str):
    """Translate huggingface_hub access failures into actionable ModelAccessErrors.

    Exception order matters: GatedRepoError subclasses RepositoryNotFoundError,
    and LocalEntryNotFoundError subclasses EntryNotFoundError.
    """
    from huggingface_hub.errors import (
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        yield
    except GatedRepoError as exc:
        raise ModelAccessError(
            f"{repo} is gated on Hugging Face; set HF_TOKEN to a token with "
            f"granted access (request it at https://huggingface.co/{repo}, "
            f"or run `hf auth login`)"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise ModelAccessError(
            f"model repo {repo!r} not found — check the name, or set HF_TOKEN "
            f"if it is private"
        ) from exc
    except LocalEntryNotFoundError as exc:
        raise ModelAccessError(
            f"{repo} is not cached locally and the network is unreachable; "
            f"connect once to download it"
        ) from exc


def _is_qlora(checkpoint: str) -> bool:
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    with _hf_access(checkpoint):
        try:
            hf_hub_download(checkpoint, "adapter_config.json")
            return True
        except LocalEntryNotFoundError:
            raise
        except EntryNotFoundError:
            return False


def _qlora_n_buckets(checkpoint: str) -> int:
    with _hf_access(checkpoint):
        path = hf_hub_download(checkpoint, "adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 — safe_open handle is not a dict
            if "score" in key and "linear.weight" in key:
                return f.get_slice(key).get_shape()[0]
    raise ValueError(f"could not infer n_buckets from adapter at {checkpoint}")


def band_for(score: float) -> str:
    """Label under the default band table; the table itself lives in
    :mod:`cx_pangram.bands`."""
    return _bands.band_for(score).label


@dataclass
class ChunkScore:
    index: int
    score: float
    bucket: int
    band: str
    n_words: int
    word_start: int
    word_end: int
    confidence: float
    truncated: bool
    probs: list[float]
    preview: str


@dataclass
class Detection:
    score: float
    band: str
    bucket: int
    n_words: int
    n_chunks: int
    reliable: bool
    confidence: float
    truncated: bool
    model: str
    calibrated: bool
    most_ai_chunk: int | None
    chunks: list[ChunkScore]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Window:
    """One model-sized slice of a document's cleaned words.

    ``word_start``/``word_end`` index into the *cleaned* word sequence — the
    text the model actually saw. ``clean_text`` is lossy by design (lowercase,
    demojize, whitespace collapse), so spans into the original string would
    need a fuzzy re-aligner; cleaned-word coordinates are the honest contract.
    """

    text: str
    n_words: int
    word_start: int
    word_end: int


@dataclass
class _Prepared:
    n_words: int
    windows: list[_Window]


@dataclass
class _WindowScore:
    """One window's forward-pass outputs, before document assembly.

    ``confidence`` is 1 − normalized entropy of the bucket distribution:
    peakedness of what the model emitted, not a probability of correctness.
    """

    score: float
    bucket: int
    probs: list[float]
    confidence: float
    truncated: bool


class EditLens:
    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        base: str | None = None,
        quantize: bool | None = None,
        bands: tuple[Band, ...] | None = None,
    ):
        self.device = select_device(device)
        if self.device.startswith("mps"):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.quantize = quantize
        self.bands = bands or _bands.BANDS
        self.model_name = model or default_model(self.device)
        if self.model_name in MODELS:
            checkpoint, default_base = MODELS[self.model_name]
        elif base:
            checkpoint, default_base = self.model_name, base
        else:
            raise ValueError(
                f"unknown model {self.model_name!r}; expected one of "
                f"{sorted(MODELS)}, or pass base= for a custom checkpoint"
            )
        base = base or default_base
        self.checkpoint = checkpoint
        self.calibrated = self.model_name == "llama"
        self.quantized = False
        with _hf_access(base):
            self.tokenizer = cast(
                PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(base)
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if _is_qlora(checkpoint):
            self.net = self._load_adapter(checkpoint, base)
            self.tokenizer.padding_side = "left"
            self.max_length = LLAMA_MAX_LENGTH
        else:
            with _hf_access(checkpoint):
                self.net = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint, dtype=_unquantized_dtype(self.device)
                )
            self.net.to(self.device)
            self.max_length = MAX_LENGTH
        # 350 words sits mid-distribution for EditLens training (75-799 words).
        # Llama's 1024-token context could take ~600-word windows for ~2x long-doc
        # throughput, but that trades unmeasured score fidelity at the training
        # distribution's edge — benchmark before changing per model.
        self.chunk_words = CHUNK_WORDS

        self.net.eval()
        self.device = str(next(self.net.parameters()).device)
        self.n_buckets = self.net.config.num_labels

    def _load_adapter(self, checkpoint: str, base: str):
        from peft import PeftModel

        n_buckets = _qlora_n_buckets(checkpoint)
        quantize = _resolve_quantize(self.device, self.quantize)
        self.quantized = quantize
        load_kwargs: dict = {"num_labels": n_buckets}
        if quantize:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            head_dtype = torch.bfloat16
            load_kwargs["dtype"] = head_dtype
            load_kwargs["device_map"] = {"": torch.device(self.device).index or 0}
        else:
            head_dtype = _unquantized_dtype(self.device)
            load_kwargs["dtype"] = head_dtype

        with _hf_access(base):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base, **load_kwargs
            )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if isinstance(getattr(base_model, "score", None), torch.nn.Linear):
            hidden = base_model.config.hidden_size
            dev = next(base_model.parameters()).device
            base_model.score = NormedLinear(
                hidden, n_buckets, device=dev, dtype=head_dtype
            )
        net = PeftModel.from_pretrained(base_model, checkpoint)
        if not quantize:
            net = net.to(device=self.device, dtype=head_dtype)
        return net

    @torch.inference_mode()
    def _score(self, texts: list[str]) -> list[_WindowScore]:
        out: list[_WindowScore] = []
        size = self._auto_batch(len(texts))
        start = 0
        while start < len(texts):
            batch = texts[start : start + size]
            try:
                scored = self._forward(batch)
            except RuntimeError as exc:
                if size == 1 or not _is_oom(exc):
                    raise
                _empty_cache(self.device)
                size = max(1, size // 2)
                continue
            out += scored
            start += len(batch)
        return out

    def _forward(self, texts: list[str]) -> list[_WindowScore]:
        raw_lengths = [
            len(ids)
            for ids in self.tokenizer(texts, truncation=False, padding=False)[
                "input_ids"
            ]
        ]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.net(**enc).logits.float()
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                f"non-finite logits from {self.model_name} on {self.device}; "
                "likely a dtype/device mismatch"
            )
        probs = torch.softmax(logits, dim=-1)
        labels = torch.arange(self.n_buckets, device=probs.device, dtype=probs.dtype)
        scores = (probs * labels).sum(-1) / (self.n_buckets - 1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
        confidence = 1.0 - entropy / math.log(self.n_buckets)
        return [
            _WindowScore(
                score=s,
                bucket=int(b),
                probs=p,
                confidence=c,
                truncated=n > self.max_length,
            )
            for s, b, p, c, n in zip(
                scores.tolist(),
                probs.argmax(-1).tolist(),
                probs.tolist(),
                confidence.tolist(),
                raw_lengths,
                strict=True,
            )
        ]

    def _auto_batch(self, n_windows: int) -> int:
        """Up-front batch width: a slice of free VRAM on CUDA (backoff in _score
        corrects overshoot), a fixed conservative width elsewhere."""
        ceiling = min(n_windows, MAX_BATCH)
        if not self.device.startswith("cuda"):
            return min(ceiling, DEFAULT_BATCH)
        free, _ = torch.cuda.mem_get_info(torch.device(self.device))
        intermediate = getattr(
            self.net.config, "intermediate_size", self.net.config.hidden_size * 4
        )
        per_window = self.max_length * intermediate * 2 * ACT_SAFETY
        est = int(free * VRAM_BUDGET / per_window)
        return max(1, min(ceiling, est))

    def _prepare(self, text: str) -> _Prepared:
        words = clean_text(text).split()
        windows = [
            _Window(
                text=" ".join(words[i : i + self.chunk_words]),
                n_words=min(self.chunk_words, len(words) - i),
                word_start=i,
                word_end=min(i + self.chunk_words, len(words)),
            )
            for i in range(0, len(words), self.chunk_words)
        ]
        return _Prepared(n_words=len(words), windows=windows)

    def _assemble(self, prep: _Prepared, scored: list[_WindowScore]) -> Detection:
        chunks = [
            ChunkScore(
                index=i,
                score=round(ws.score, 4),
                bucket=ws.bucket,
                band=_bands.band_for(ws.score, self.bands).label,
                n_words=w.n_words,
                word_start=w.word_start,
                word_end=w.word_end,
                confidence=round(ws.confidence, 4),
                truncated=ws.truncated,
                probs=[round(x, 4) for x in ws.probs],
                preview=w.text[:160],
            )
            for i, (w, ws) in enumerate(zip(prep.windows, scored, strict=True))
        ]
        reliable = prep.n_words >= MIN_WORDS
        if not chunks:
            return Detection(
                score=0.0,
                band="unreliable",
                bucket=0,
                n_words=0,
                n_chunks=0,
                reliable=False,
                confidence=0.0,
                truncated=False,
                model=self.model_name,
                calibrated=self.calibrated,
                most_ai_chunk=None,
                chunks=[],
            )
        total = sum(c.n_words for c in chunks) or 1
        agg = sum(c.score * c.n_words for c in chunks) / total
        conf = sum(c.confidence * c.n_words for c in chunks) / total
        most_ai = max(chunks, key=lambda c: c.score).index
        return Detection(
            score=round(agg, 4),
            band=_bands.band_for(agg, self.bands).label if reliable else "unreliable",
            bucket=round(agg * (self.n_buckets - 1)),
            n_words=prep.n_words,
            n_chunks=len(chunks),
            reliable=reliable,
            confidence=round(conf, 4),
            truncated=any(c.truncated for c in chunks),
            model=self.model_name,
            calibrated=self.calibrated,
            most_ai_chunk=most_ai,
            chunks=chunks,
        )

    def detect_batch(self, texts: Sequence[str]) -> list[Detection]:
        """Score many documents in one flat, VRAM-aware pass.

        Windows from all documents are batched together, so many short
        documents fill batches a per-document loop would leave mostly empty.
        Windows stay contiguous per document; a cursor walk regroups them.
        """
        preps = [self._prepare(t) for t in texts]
        flat = [w.text for p in preps for w in p.windows]
        scored = self._score(flat) if flat else []
        out: list[Detection] = []
        cursor = 0
        for prep in preps:
            k = len(prep.windows)
            out.append(self._assemble(prep, scored[cursor : cursor + k]))
            cursor += k
        return out

    def detect(self, text: str) -> Detection:
        return self.detect_batch([text])[0]
