"""Verify _drop_untradable strips delisted / non-equity rows from the listing
DataFrame, regardless of broker-specific column-value casing."""
import pandas as pd

from stockpredict.data.universe import _drop_untradable


def test_vci_drops_delisted_and_non_equity():
    """VCI's symbols_by_exchange() returns exchange={HSX,HNX,UPCOM,DELISTED,BOND}
    and type={STOCK,ETF,CW,FU,BOND,DEBENTURE,UNIT_TRUST}. After filtering we
    keep only tradable equities/ETFs on a real exchange — HTK in particular
    must drop out."""
    raw = pd.DataFrame([
        {"symbol": "VCB", "exchange": "HSX",      "type": "STOCK"},
        {"symbol": "AGX", "exchange": "UPCOM",    "type": "STOCK"},
        {"symbol": "DST", "exchange": "HNX",      "type": "STOCK"},
        {"symbol": "FUEVFVND", "exchange": "HSX", "type": "ETF"},
        {"symbol": "HTK", "exchange": "DELISTED", "type": "STOCK"},
        {"symbol": "DEF", "exchange": "DELISTED", "type": "CW"},
        {"symbol": "CFPT2502", "exchange": "HSX", "type": "CW"},
        {"symbol": "VN30F2506", "exchange": "HNX", "type": "FU"},
        {"symbol": "BOND123", "exchange": "BOND", "type": "BOND"},
        {"symbol": "XYZ", "exchange": "HSX",      "type": "DEBENTURE"},
    ])
    out = _drop_untradable(raw)
    keep = set(out["symbol"])
    assert keep == {"VCB", "AGX", "DST", "FUEVFVND"}
    assert "HTK" not in keep, "delisted ticker leaked through"


def test_kbs_drops_lowercase_non_equity():
    """KBS uses HOSE (not HSX) and lowercase types like 'cw' / 'corpbond' /
    'future'. Drop the non-equity rows but keep stocks and fund-class rows."""
    raw = pd.DataFrame([
        {"symbol": "VCB", "exchange": "HOSE",  "type": "stock"},
        {"symbol": "BSR", "exchange": "HOSE",  "type": "stock"},
        {"symbol": "ACV", "exchange": "UPCOM", "type": "stock"},
        {"symbol": "FUEVFVND", "exchange": "HOSE", "type": "fund"},
        {"symbol": "CFPT2502", "exchange": "HOSE", "type": "cw"},
        {"symbol": "BOND456", "exchange": "HOSE",  "type": "corpbond"},
        {"symbol": "F2506",   "exchange": "XHNF",  "type": "future"},
    ])
    out = _drop_untradable(raw)
    keep = set(out["symbol"])
    assert keep == {"VCB", "BSR", "ACV", "FUEVFVND"}


def test_noop_when_neither_exchange_nor_type_present():
    """The legacy ``all_symbols()`` payload has only [symbol, organ_name].
    With no filterable columns, _drop_untradable must pass everything
    through — the warm-cache layer of the selector picks up the slack."""
    raw = pd.DataFrame([
        {"symbol": "VCB", "organ_name": "Vietcombank"},
        {"symbol": "HTK", "organ_name": None},
    ])
    out = _drop_untradable(raw)
    assert list(out["symbol"]) == ["VCB", "HTK"]
