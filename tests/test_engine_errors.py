"""Access-error translation, unknown-model rejection, and OOM classification."""

import httpx
import pytest
import torch
from huggingface_hub.errors import (
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)

from cx_pangram import ModelAccessError
from cx_pangram.engine import EditLens, _hf_access, _is_oom


def _http_response() -> httpx.Response:
    return httpx.Response(401, request=httpx.Request("GET", "https://hf.co/x"))


def test_gated_repo_becomes_actionable_error():
    with (
        pytest.raises(ModelAccessError, match="HF_TOKEN"),
        _hf_access("pangram/editlens_Llama-3.2-3B"),
    ):
        raise GatedRepoError("401", response=_http_response())


def test_missing_repo_names_the_repo():
    with (
        pytest.raises(ModelAccessError, match="not found"),
        _hf_access("no/such-repo"),
    ):
        raise RepositoryNotFoundError("404", response=_http_response())


def test_offline_uncached_mentions_network():
    with (
        pytest.raises(ModelAccessError, match="network"),
        _hf_access("pangram/editlens_roberta-large"),
    ):
        raise LocalEntryNotFoundError("offline")


def test_unknown_model_without_base_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        EditLens(model="not-a-model", device="cpu")


def test_is_oom_covers_cuda_and_mps():
    assert _is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert _is_oom(RuntimeError("MPS backend out of memory (MPS allocated ...)"))
    assert not _is_oom(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))


def test_empty_text_reads_unreliable():
    eng = object.__new__(EditLens)
    eng.model_name = "stub"
    eng.calibrated = False
    det = eng.detect("   \n\t  ")
    assert det.band == "unreliable"
    assert det.reliable is False
    assert det.n_words == 0 and det.chunks == []
