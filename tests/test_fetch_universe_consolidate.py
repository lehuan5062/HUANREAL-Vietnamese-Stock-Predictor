"""Verify fetch_universe() consolidates both KBS and VCI sources correctly."""
import pandas as pd
from unittest.mock import Mock, patch
import pytest

from stockpredict.data.universe import fetch_universe


def test_fetch_universe_consolidates_both_sources(monkeypatch):
    """When both KBS and VCI sources succeed, consolidate them."""
    kbs_data = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE"},
        {"symbol": "TCB", "organ_name": "Techcombank", "exchange": "HOSE"},
    ])
    vci_data = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "VIETCOMBANK", "exchange": "HSX"},
        {"symbol": "HDB", "organ_name": "HDBank", "exchange": "HSX"},
    ])

    def mock_listing_init(source):
        mock_listing = Mock()
        if source == "KBS":
            # KBS returns via symbols_by_exchange (which gets normalized)
            mock_listing.symbols_by_exchange = lambda: kbs_data.copy()
        else:  # VCI
            mock_listing.symbols_by_exchange = lambda: vci_data.copy()
        mock_listing.all_symbols = Mock(return_value=pd.DataFrame())
        mock_listing.all_etf = Mock(return_value=None)
        return mock_listing

    # Mock vnstock.Listing (imported lazily in fetch_universe)
    with patch("vnstock.Listing", side_effect=mock_listing_init):
        with patch("stockpredict.data.universe._try_fetch_etf_listing", return_value=None):
            result = fetch_universe()

    # Should consolidate both sources
    assert len(result) == 3, "Should have 3 unique symbols (VCB, TCB, HDB)"
    assert set(result["symbol"]) == {"VCB", "TCB", "HDB"}
    assert result["source"].iloc[0] == "KBS+VCI", "Source should be marked as consolidated"
    # No duplicates
    assert len(result) == len(result["symbol"].unique())


def test_fetch_universe_uses_single_source_when_one_fails(monkeypatch):
    """When one source fails and the other succeeds, use the successful one."""
    kbs_data = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE"},
        {"symbol": "TCB", "organ_name": "Techcombank", "exchange": "HOSE"},
    ])

    def mock_listing_init(source):
        mock_listing = Mock()
        if source == "KBS":
            mock_listing.symbols_by_exchange = lambda: kbs_data.copy()
        else:  # VCI
            # VCI fails
            mock_listing.symbols_by_exchange = Mock(side_effect=Exception("VCI failed"))
        mock_listing.all_symbols = Mock(side_effect=Exception("Failed"))
        mock_listing.all_etf = Mock(return_value=None)
        return mock_listing

    with patch("vnstock.Listing", side_effect=mock_listing_init):
        with patch("stockpredict.data.universe._try_fetch_etf_listing", return_value=None):
            # VCI will fail, but KBS will succeed
            result = fetch_universe()

    assert len(result) == 2
    assert set(result["symbol"]) == {"VCB", "TCB"}
    assert result["source"].iloc[0] == "KBS", "Should use KBS since VCI failed"


def test_fetch_universe_raises_when_both_fail(monkeypatch):
    """When both sources fail, raise RuntimeError."""
    def mock_listing_init(source):
        mock_listing = Mock()
        mock_listing.symbols_by_exchange = Mock(side_effect=Exception(f"{source} failed"))
        mock_listing.all_symbols = Mock(side_effect=Exception(f"{source} failed"))
        return mock_listing

    with patch("vnstock.Listing", side_effect=mock_listing_init):
        with pytest.raises(RuntimeError, match="All vnstock sources failed"):
            fetch_universe()


def test_fetch_universe_merges_etfs(monkeypatch):
    """ETF list from KBS is merged into the consolidated stock listing."""
    kbs_data = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE", "instrument_type": "STOCK"},
    ])
    vci_data = pd.DataFrame([
        {"symbol": "FUEVFVND", "organ_name": "Diamond VN30", "exchange": "HSX", "instrument_type": "ETF"},
    ])
    etf_data = pd.DataFrame([
        {"symbol": "FUEVFVND", "instrument_type": "ETF"},
        {"symbol": "E1VFVN30", "instrument_type": "ETF"},
    ])

    def mock_listing_init(source):
        mock_listing = Mock()
        if source == "KBS":
            mock_listing.symbols_by_exchange = lambda: kbs_data.copy()
        else:  # VCI
            mock_listing.symbols_by_exchange = lambda: vci_data.copy()
        mock_listing.all_symbols = Mock(return_value=pd.DataFrame())
        return mock_listing

    with patch("vnstock.Listing", side_effect=mock_listing_init):
        with patch("stockpredict.data.universe._try_fetch_etf_listing", return_value=etf_data):
            result = fetch_universe()

    # Should have consolidated stocks + merged ETF list
    assert "FUEVFVND" in result["symbol"].values
    assert "E1VFVN30" in result["symbol"].values
    assert "VCB" in result["symbol"].values
