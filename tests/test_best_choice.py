"""Verify the multi-category 'best choice' badges are applied correctly."""
import pandas as pd

from stockpredict.picks_meta import annotate_best


def _frame(rows):
    return pd.DataFrame(rows)


def test_no_actionable_no_badges():
    """When no row is actionable, no badges anywhere."""
    df = _frame([
        {"symbol": "A", "actionable": False, "adjusted": 0.01,
         "rr_ratio": 0.5, "net_reward_vnd": 1000},
        {"symbol": "B", "actionable": False, "adjusted": 0.02,
         "rr_ratio": 0.3, "net_reward_vnd": 2000},
    ])
    out = annotate_best(df)
    for col in ("best_adjusted", "best_rr", "best_net", "best_composite"):
        assert col in out.columns
        assert not out[col].any(), f"{col} should be all False"


def test_single_actionable_gets_all_four_badges():
    """A lone actionable row sweeps all four categories."""
    df = _frame([
        {"symbol": "A", "actionable": False, "adjusted": 0.05,
         "rr_ratio": 5.0, "net_reward_vnd": 99999},
        {"symbol": "B", "actionable": True, "adjusted": 0.01,
         "rr_ratio": 0.9, "net_reward_vnd": 1000},
    ])
    out = annotate_best(df)
    b_row = out[out["symbol"] == "B"].iloc[0]
    assert b_row["best_adjusted"]
    assert b_row["best_rr"]
    assert b_row["best_net"]
    assert b_row["best_composite"]
    a_row = out[out["symbol"] == "A"].iloc[0]
    for col in ("best_adjusted", "best_rr", "best_net", "best_composite"):
        assert not a_row[col], f"non-actionable A should not be flagged on {col}"


def test_multiple_actionable_per_category_leaders():
    """Each category picks its own leader; a ticker can win multiple."""
    df = _frame([
        # A: highest rr_ratio                 — should win best_rr
        {"symbol": "A", "actionable": True,  "adjusted": 0.02,
         "rr_ratio": 2.0, "net_reward_vnd": 5000},
        # B: highest adjusted AND highest net — should win best_adjusted + best_net
        {"symbol": "B", "actionable": True,  "adjusted": 0.05,
         "rr_ratio": 1.0, "net_reward_vnd": 99999},
        # C: middle of the pack on everything
        {"symbol": "C", "actionable": True,  "adjusted": 0.03,
         "rr_ratio": 1.5, "net_reward_vnd": 7000},
        # D: not actionable, must NEVER be flagged
        {"symbol": "D", "actionable": False, "adjusted": 0.99,
         "rr_ratio": 99.0, "net_reward_vnd": 9_999_999},
    ])
    out = annotate_best(df)
    by_sym = {r["symbol"]: r for _, r in out.iterrows()}
    assert by_sym["A"]["best_rr"]
    assert not by_sym["A"]["best_adjusted"]
    assert not by_sym["A"]["best_net"]
    assert by_sym["B"]["best_adjusted"]
    assert by_sym["B"]["best_net"]
    assert not by_sym["B"]["best_rr"]
    # composite leader: smallest sum of ranks across all 3.
    # A ranks: adj=3, rr=1, net=3 -> 7
    # B ranks: adj=1, rr=3, net=1 -> 5  (winner)
    # C ranks: adj=2, rr=2, net=2 -> 6
    assert by_sym["B"]["best_composite"]
    assert not by_sym["A"]["best_composite"]
    assert not by_sym["C"]["best_composite"]
    # D never flagged (not actionable)
    for col in ("best_adjusted", "best_rr", "best_net", "best_composite"):
        assert not by_sym["D"][col]


def test_idempotent():
    """Calling annotate_best twice should yield the same result."""
    df = _frame([
        {"symbol": "A", "actionable": True, "adjusted": 0.05,
         "rr_ratio": 1.5, "net_reward_vnd": 10000},
        {"symbol": "B", "actionable": True, "adjusted": 0.03,
         "rr_ratio": 2.0, "net_reward_vnd": 5000},
    ])
    a = annotate_best(df)
    b = annotate_best(a)
    assert a.equals(b)


def test_missing_actionable_column_is_safe():
    """If the picks frame has no `actionable` column, no flags are set —
    we don't infer it."""
    df = _frame([
        {"symbol": "A", "adjusted": 0.05, "rr_ratio": 1.5, "net_reward_vnd": 10000},
    ])
    out = annotate_best(df)
    for col in ("best_adjusted", "best_rr", "best_net", "best_composite"):
        assert col in out.columns
        assert not out[col].any()


def test_empty_input_returns_empty():
    """Empty frame in -> empty frame out, no crash."""
    out = annotate_best(pd.DataFrame())
    assert out is not None
    assert len(out) == 0
