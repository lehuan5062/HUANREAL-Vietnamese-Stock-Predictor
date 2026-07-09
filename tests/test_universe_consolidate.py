"""Verify _merge_stock_listings() consolidates KBS and VCI stock listings correctly."""
import pandas as pd

from stockpredict.data.universe import _merge_stock_listings


def test_merge_identical_symbols():
    """Both sources have the same symbols. Consolidation should deduplicate
    and prefer the more complete row."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE"},
        {"symbol": "TCB", "organ_name": "Techcombank", "exchange": "HOSE"},
    ])
    vci = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "VIETCOMBANK", "exchange": "HSX"},
        {"symbol": "TCB", "organ_name": None, "exchange": "HSX"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 2, "Should have 2 unique symbols"
    assert "VCB" in result["symbol"].values
    assert "TCB" in result["symbol"].values
    # No duplicates
    assert len(result) == len(result["symbol"].unique())


def test_merge_different_symbols():
    """KBS and VCI have different symbols. Consolidation should union all."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE"},
        {"symbol": "TCB", "organ_name": "Techcombank", "exchange": "HOSE"},
    ])
    vci = pd.DataFrame([
        {"symbol": "HDB", "organ_name": "HDBank", "exchange": "HSX"},
        {"symbol": "ACB", "organ_name": "ACB Bank", "exchange": "HSX"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 4, "Should have 4 unique symbols from union"
    assert set(result["symbol"]) == {"VCB", "TCB", "HDB", "ACB"}


def test_merge_overlapping_symbols():
    """KBS and VCI have some overlapping and some unique symbols."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE"},
        {"symbol": "TCB", "organ_name": "Techcombank", "exchange": "HOSE"},
        {"symbol": "BID", "organ_name": None, "exchange": None},
    ])
    vci = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "VIETCOMBANK", "exchange": "HSX"},
        {"symbol": "HDB", "organ_name": "HDBank", "exchange": "HSX"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 4, "Should have 4 unique symbols"
    assert set(result["symbol"]) == {"VCB", "TCB", "BID", "HDB"}


def test_merge_prefer_more_complete_row():
    """When a symbol exists in both sources, prefer the row with fewer nulls."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": "HOSE", "type": None},
    ])
    vci = pd.DataFrame([
        {"symbol": "VCB", "organ_name": None, "exchange": "HSX", "type": "STOCK"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 1
    row = result.iloc[0]
    # VCI has fewer nulls (2 vs 3 for KBS), so use VCI as base and fill from KBS
    assert row["symbol"] == "VCB"
    # Should have all non-null fields from both sources
    assert row["organ_name"] == "Vietcombank"  # filled from KBS
    assert row["type"] == "STOCK"  # from VCI


def test_merge_tied_null_count_prefers_vci():
    """When both rows have same number of nulls, prefer VCI."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "exchange": None},
    ])
    vci = pd.DataFrame([
        {"symbol": "VCB", "organ_name": None, "exchange": "HSX"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 1
    row = result.iloc[0]
    # Both have 1 null; prefer VCI as base
    assert row["symbol"] == "VCB"
    assert row["exchange"] == "HSX"  # from VCI
    assert row["organ_name"] == "Vietcombank"  # filled from KBS


def test_merge_empty_dataframes():
    """Handle empty DataFrames gracefully."""
    kbs = pd.DataFrame()
    vci = pd.DataFrame([{"symbol": "VCB", "organ_name": "Vietcombank"}])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "VCB"

    # Reverse: KBS has data, VCI is empty
    result = _merge_stock_listings(vci, kbs)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "VCB"


def test_merge_none_dataframes():
    """Handle None DataFrames gracefully."""
    kbs = None
    vci = pd.DataFrame([{"symbol": "VCB", "organ_name": "Vietcombank"}])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "VCB"

    # Reverse
    result = _merge_stock_listings(vci, None)
    assert len(result) == 1


def test_merge_reclassifies_instrument_type():
    """After consolidation, re-classify instrument types using symbol patterns."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank", "instrument_type": "STOCK"},
        {"symbol": "FUEVFVND", "organ_name": "Diamond VN30", "instrument_type": "FUND"},
    ])
    vci = pd.DataFrame([
        {"symbol": "VCB", "organ_name": None, "instrument_type": "STOCK"},
        {"symbol": "E1VFVN30", "organ_name": "Legacy VN30 ETF", "instrument_type": "OTHER"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 3
    # Symbols should be re-classified by pattern
    vcb_row = result[result["symbol"] == "VCB"].iloc[0]
    assert vcb_row["instrument_type"] == "STOCK"  # 3 letters = STOCK
    fue_row = result[result["symbol"] == "FUEVFVND"].iloc[0]
    assert fue_row["instrument_type"] == "ETF"  # FUE* = ETF
    e1v_row = result[result["symbol"] == "E1VFVN30"].iloc[0]
    assert e1v_row["instrument_type"] == "ETF"  # E1VFVN30 = ETF


def test_merge_case_insensitive_symbol_matching():
    """Symbol matching should be case-insensitive."""
    kbs = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank"},
    ])
    vci = pd.DataFrame([
        {"symbol": "vcb", "organ_name": "Vietcombank Lower"},
    ])
    result = _merge_stock_listings(kbs, vci)
    assert len(result) == 1, "Should have 1 unique symbol despite case difference"
    assert result.iloc[0]["symbol"] == "VCB"  # normalized to uppercase
