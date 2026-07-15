"""One-off cleaner for the rebound_config_tuner results JSONL.

Removes any corrupted (non-JSON-parseable) lines from
reports/tuning/rebound_include_held_search.jsonl. Run this only when
run_config_suggest.bat prints a "skipped N corrupted line(s)" warning — the
tuner's write-time newline guard makes corruption rare, and the suggest script
already tolerates bad lines, so this is just housekeeping.

Safe by construction:
  1. Backs the original file up to reports/tuning/config_backups/ first.
  2. Writes the cleaned content to a temp file in the same directory, then
     atomically os.replace()s it into place — the clean pass itself can never
     leave the results file half-written.

After cleaning, the file is smaller, so rebound_config_suggest's incremental
parse cache auto-invalidates (its cached byte offset exceeds the new file size)
and does a full re-parse on the next run. No manual cache deletion needed.

    python -m scripts.rebound_clean_results
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from stockpredict import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning" / "rebound_include_held_search.jsonl"
BACKUP_DIR = PROJECT_ROOT / "reports" / "tuning" / "config_backups"


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"No results file at {RESULTS_PATH} — nothing to clean.")
        return

    raw = RESULTS_PATH.read_bytes()
    lines = raw.decode("utf-8", errors="replace").splitlines()

    good, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        good.append(line)

    if bad == 0:
        print(f"{len(good)} lines, all valid — nothing to clean.")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"rebound_include_held_search_{timestamp}.jsonl.bak"
    backup_path.write_bytes(raw)

    tmp_path = RESULTS_PATH.with_suffix(".jsonl.tmp")
    tmp_path.write_text("\n".join(good) + "\n", encoding="utf-8")
    os.replace(tmp_path, RESULTS_PATH)

    print(f"Dropped {bad} corrupted line(s); kept {len(good)}.")
    print(f"Original backed up to {backup_path}")
    print(f"Cleaned {RESULTS_PATH}")


if __name__ == "__main__":
    main()
