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

import os
from dataclasses import asdict, dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocess import clean_text, count_words

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
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


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


def _is_qlora(checkpoint: str) -> bool:
    try:
        hf_hub_download(checkpoint, "adapter_config.json")
        return True
    except Exception:
        return False


def _qlora_n_buckets(checkpoint: str) -> int:
    path = hf_hub_download(checkpoint, "adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():  # noqa: SIM118 — safe_open handle is not a dict
            if "score" in key and "linear.weight" in key:
                return f.get_slice(key).get_shape()[0]
    raise ValueError(f"could not infer n_buckets from adapter at {checkpoint}")


_BANDS = (
    (0.30, "human"),
    (0.55, "lightly edited"),
    (0.75, "moderately edited"),
    (0.90, "heavily edited"),
)


def band_for(score: float) -> str:
    for hi, name in _BANDS:
        if score < hi:
            return name
    return "fully AI"


@dataclass
class ChunkScore:
    index: int
    score: float
    bucket: int
    band: str
    n_words: int
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
    model: str
    calibrated: bool
    most_ai_chunk: int | None
    chunks: list[ChunkScore]

    def to_dict(self) -> dict:
        return asdict(self)


class EditLens:
    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        base: str | None = None,
        quantize: bool | None = None,
    ):
        self.device = select_device(device)
        if self.device.startswith("mps"):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.quantize = quantize
        self.model_name = model or default_model(self.device)
        checkpoint, default_base = MODELS.get(
            self.model_name, (self.model_name, MODELS["roberta"][1])
        )
        base = base or default_base
        self.checkpoint = checkpoint
        self.calibrated = self.model_name == "llama"
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if _is_qlora(checkpoint):
            self.net = self._load_adapter(checkpoint, base)
            self.tokenizer.padding_side = "left"
            self.max_length = LLAMA_MAX_LENGTH
        else:
            self.net = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, dtype=_unquantized_dtype(self.device)
            )
            self.net.to(self.device)
            self.max_length = MAX_LENGTH

        self.net.eval()
        self.device = str(next(self.net.parameters()).device)
        self.n_buckets = self.net.config.num_labels

    def _load_adapter(self, checkpoint: str, base: str):
        from peft import PeftModel

        n_buckets = _qlora_n_buckets(checkpoint)
        quantize = _resolve_quantize(self.device, self.quantize)
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

    @torch.no_grad()
    def _score(
        self, texts: list[str]
    ) -> tuple[list[float], list[int], list[list[float]]]:
        scores: list[float] = []
        buckets: list[int] = []
        probs: list[list[float]] = []
        size = self._auto_batch(len(texts))
        start = 0
        while start < len(texts):
            batch = texts[start : start + size]
            try:
                s, b, p = self._forward(batch)
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size = max(1, size // 2)
                continue
            scores += s
            buckets += b
            probs += p
            start += len(batch)
        return scores, buckets, probs

    def _forward(
        self, texts: list[str]
    ) -> tuple[list[float], list[int], list[list[float]]]:
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
        return scores.tolist(), probs.argmax(-1).tolist(), probs.tolist()

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

    def detect(self, text: str) -> Detection:
        cleaned = clean_text(text)
        words = cleaned.split()
        n_words = count_words(cleaned)
        if not words:
            return Detection(
                0.0,
                band_for(0.0),
                0,
                0,
                0,
                False,
                self.model_name,
                self.calibrated,
                None,
                [],
            )

        windows = [
            " ".join(words[i : i + CHUNK_WORDS])
            for i in range(0, len(words), CHUNK_WORDS)
        ]
        scores, buckets, probs = self._score(windows)
        chunks = [
            ChunkScore(
                index=i,
                score=round(s, 4),
                bucket=int(b),
                band=band_for(s),
                n_words=len(w.split()),
                probs=[round(x, 4) for x in p],
                preview=w[:160],
            )
            for i, (w, s, b, p) in enumerate(
                zip(windows, scores, buckets, probs, strict=True)
            )
        ]
        total = sum(c.n_words for c in chunks) or 1
        agg = sum(c.score * c.n_words for c in chunks) / total
        most_ai = max(chunks, key=lambda c: c.score).index
        reliable = n_words >= MIN_WORDS
        return Detection(
            score=round(agg, 4),
            band=band_for(agg) if reliable else "unreliable",
            bucket=round(agg * (self.n_buckets - 1)),
            n_words=n_words,
            n_chunks=len(chunks),
            reliable=reliable,
            model=self.model_name,
            calibrated=self.calibrated,
            most_ai_chunk=most_ai,
            chunks=chunks,
        )
