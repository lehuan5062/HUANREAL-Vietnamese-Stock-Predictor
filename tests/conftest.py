"""Make the src/ layout importable without installing the package."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path_factory, monkeypatch):
    """Redirect the predictions ledger to a throwaway temp file for EVERY test.

    Several tests call ``tracking.record`` (directly or via ``base.run`` / a mode
    ``finalize``) with fake fixture symbols. ``ledger_path()`` is the single I/O
    chokepoint for ``_read`` / ``_write``, so pointing it at a temp file keeps
    those writes out of the real ``cache/predictions.parquet`` (which they had
    been polluting with rows like AAA/BBB/CCC and impossible returns)."""
    from stockpredict import tracking
    led = tmp_path_factory.mktemp("ledger") / "predictions.parquet"
    monkeypatch.setattr(tracking, "ledger_path", lambda: led)
    yield
