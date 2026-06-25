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


class NormedLinear(torch.nn.Module):
    """LayerNorm + bias-free Linear; EditLens's classification head on causal backbones.

    Ported verbatim from pangramlabs/EditLens (train.py). The Llama adapter saves this
    head via LoRA `modules_to_save`, so the architecture must match before loading.
    """

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
        for key in f.keys():
            if "score" in key and "linear.weight" in key:
                return f.get_slice(key).get_shape()[0]
    raise ValueError(f"could not infer n_buckets from adapter at {checkpoint}")


# Calibrated for the llama-3.2-3B backbone (the default) on a human-vs-AI Are.na
# corpus: lightly/moderately (0.55) sits just above the confirmed-human max (0.512)
# so human prose never reads as moderately+. The >=0.55 bands are headroom only a
# stronger signal (the unreleased 24B) trips; roberta over-flags and reads high here.
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
    most_ai_chunk: int | None
    chunks: list[ChunkScore]

    def to_dict(self) -> dict:
        return asdict(self)


class EditLens:
    def __init__(
        self, model: str = "llama", device: str | None = None, base: str | None = None
    ):
        checkpoint, default_base = MODELS.get(model, (model, MODELS["roberta"][1]))
        base = base or default_base
        self.model_name = model
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if _is_qlora(checkpoint):
            self.net, self.device = self._load_qlora(checkpoint, base)
            self.tokenizer.padding_side = "left"
            self.max_length = LLAMA_MAX_LENGTH
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.net = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            self.net.to(self.device)
            self.max_length = MAX_LENGTH

        self.net.eval()
        self.n_buckets = self.net.config.num_labels

    def _load_qlora(self, checkpoint: str, base: str):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "llama backbone needs CUDA (4-bit quantization); none found"
            )
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        n_buckets = _qlora_n_buckets(checkpoint)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=n_buckets, quantization_config=quant, device_map={"": 0}
        )
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if isinstance(getattr(base_model, "score", None), torch.nn.Linear):
            hidden = base_model.config.hidden_size
            dev = next(base_model.parameters()).device
            base_model.score = NormedLinear(
                hidden, n_buckets, device=dev, dtype=torch.bfloat16
            )
        net = PeftModel.from_pretrained(base_model, checkpoint)
        return net, next(net.parameters()).device

    @torch.no_grad()
    def _score(
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
        probs = torch.softmax(logits, dim=-1)
        labels = torch.arange(self.n_buckets, device=probs.device, dtype=probs.dtype)
        scores = (probs * labels).sum(-1) / (self.n_buckets - 1)
        return scores.tolist(), probs.argmax(-1).tolist(), probs.tolist()

    def detect(self, text: str) -> Detection:
        cleaned = clean_text(text)
        words = cleaned.split()
        n_words = count_words(cleaned)
        if not words:
            return Detection(
                0.0, band_for(0.0), 0, 0, 0, False, self.model_name, None, []
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
            for i, (w, s, b, p) in enumerate(zip(windows, scores, buckets, probs))
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
            most_ai_chunk=most_ai,
            chunks=chunks,
        )
