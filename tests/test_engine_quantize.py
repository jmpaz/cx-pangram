"""engine._resolve_quantize maps a quantize preference onto a 4-bit decision."""

import pytest

from cx_pangram import engine


def test_none_defers_to_bnb_usable(monkeypatch):
    monkeypatch.setattr(engine, "_bnb_usable", lambda device: True)
    assert engine._resolve_quantize("cuda:0", None) is True
    monkeypatch.setattr(engine, "_bnb_usable", lambda device: False)
    assert engine._resolve_quantize("cuda:0", None) is False


def test_false_always_unquantized(monkeypatch):
    monkeypatch.setattr(engine, "_bnb_usable", lambda device: True)
    assert engine._resolve_quantize("cuda:0", False) is False
    assert engine._resolve_quantize("cpu", False) is False


def test_true_requires_bnb(monkeypatch):
    monkeypatch.setattr(engine, "_bnb_usable", lambda device: True)
    assert engine._resolve_quantize("cuda:0", True) is True


def test_true_without_bnb_raises(monkeypatch):
    monkeypatch.setattr(engine, "_bnb_usable", lambda device: False)
    with pytest.raises(RuntimeError):
        engine._resolve_quantize("cpu", True)


def test_bnb_unusable_off_cuda_without_monkeypatch():
    assert engine._resolve_quantize("cpu", None) is False
    assert engine._resolve_quantize("mps", None) is False
    with pytest.raises(RuntimeError):
        engine._resolve_quantize("cpu", True)
