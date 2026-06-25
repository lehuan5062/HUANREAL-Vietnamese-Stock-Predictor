"""Tests for the overbought hard gate (RSI-only exclusion).

An overbought blow-off (price run too far) tends to reverse, so buying the top
is a poor T+2 entry. overbought_mask drops candidates whose rsi_14 exceeds
pricing.overbought_rsi_max (0 = off). Distinct from the liquidity volume-spike
defense (min_adv_active_days), which guards tradability, not exhaustion.
"""
from __future__ import annotations

import pandas as pd

from stockpredict import filters


def _patch_level(monkeypatch, level):
    """Patch pricing.overbought_rsi_max via a fake config object."""
    class _Cfg:
        pricing = {"overbought_rsi_max": level}
    monkeypatch.setattr(filters, "load_config", lambda: _Cfg())


def _mask(df):
    return filters.overbought_mask(df)


def test_disabled_by_default(monkeypatch):
    """Level 0 -> gate off, every row kept."""
    _patch_level(monkeypatch, 0)
    df = pd.DataFrame({"rsi_14": [50.0, 95.0]})
    assert _mask(df).tolist() == [True, True]


def test_excludes_above_level(monkeypatch):
    """rsi_14 strictly above the level is dropped; below is kept."""
    _patch_level(monkeypatch, 70)
    df = pd.DataFrame({"rsi_14": [55.0, 85.0]})
    assert _mask(df).tolist() == [True, False]


def test_boundary_is_kept(monkeypatch):
    """rsi_14 exactly == level is kept (strict >)."""
    _patch_level(monkeypatch, 80)
    df = pd.DataFrame({"rsi_14": [80.0]})
    assert _mask(df).iloc[0] is True or bool(_mask(df).iloc[0]) is True


def test_missing_column_passthrough(monkeypatch):
    """No rsi_14 column -> can't gate -> all kept."""
    _patch_level(monkeypatch, 70)
    df = pd.DataFrame({"close": [10.0, 20.0]})
    assert _mask(df).tolist() == [True, True]


def test_nan_rsi_kept(monkeypatch):
    """NaN RSI is not 'overbought' -> kept."""
    _patch_level(monkeypatch, 70)
    df = pd.DataFrame({"rsi_14": [float("nan"), 90.0]})
    assert _mask(df).tolist() == [True, False]
