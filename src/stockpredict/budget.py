"""Time-budget planner: convert minutes-before-market-open into a universe size.

The vnstock guest tier rate-limits at 20 requests/minute. We model:
  - 1 vnstock API call per ticker (incremental or full history)
  - already-up-to-date cached tickers cost 0 API calls (we skip them)
  - reserve some minutes for training + prediction + (optional) news work
"""
from __future__ import annotations

from dataclasses import dataclass


# Empirically: vnstock guest = 20 req/min. We aim slightly under to leave headroom.
SAFE_API_PER_MIN = 18.0

# Fixed pipeline overhead beyond data fetching (engineer features, train, predict, write).
OVERHEAD_MIN_BASE = 3.0       # base mode -> emit picks
OVERHEAD_MIN_GEMINI = 3.5     # base + emit Gemini prompt file
OVERHEAD_MIN_CLAUDE = 8.0     # base + emit plan + caller does ~20 WebFetches


def overhead_for(mode: str) -> float:
    return {
        "base": OVERHEAD_MIN_BASE,
        "gemini": OVERHEAD_MIN_GEMINI,
        "claude": OVERHEAD_MIN_CLAUDE,
    }.get(mode, OVERHEAD_MIN_BASE)


@dataclass
class RunPlan:
    duration_min: int
    mode: str
    overhead_min: float
    api_per_min: float
    universe_target: int           # how many tickers to include in the run
    api_call_budget: int           # how many vnstock fetches we can afford

    def summary(self) -> str:
        if self.duration_min < 0:
            dur = "FULL (no time cap)"
        else:
            dur = f"{self.duration_min} min"
        return (
            f"duration={dur}  mode={self.mode}\n"
            f"  overhead reserved: {self.overhead_min:.1f} min "
            f"(features + train + predict + emit)\n"
            f"  fetch budget: {self.api_call_budget} API calls "
            f"@ {self.api_per_min}/min\n"
            f"  target universe size: {self.universe_target} tickers"
        )


def plan(duration_min: int | str, mode: str = "base",
         min_universe: int = 50,
         max_universe: int = 1600) -> RunPlan:
    """Return a RunPlan describing how to use the budget.

    Pass `duration_min="full"` (or `None`) to run on the entire universe with no
    time cap — useful weekly / monthly when you have all the time you want and
    can wait for the full ~1,500-ticker fetch to finish.

    Otherwise the universe target is the number of tickers we keep in the run.
    Tickers already cached and up-to-date cost no API calls, so we may end up
    *able* to include more than `api_call_budget` if the cache is warm.
    For sizing decisions we use api_call_budget as a hard cap on *new* fetches.
    """
    overhead = overhead_for(mode)
    if duration_min is None or (isinstance(duration_min, str) and duration_min.lower() == "full"):
        return RunPlan(
            duration_min=-1,
            mode=mode,
            overhead_min=overhead,
            api_per_min=SAFE_API_PER_MIN,
            universe_target=max_universe,
            api_call_budget=max_universe,
        )
    duration_int = int(duration_min)
    available = max(0.0, duration_int - overhead)
    api_calls = int(available * SAFE_API_PER_MIN)
    # We always want at least min_universe tickers, even if it means going over.
    target = max(min_universe, api_calls)
    target = min(target, max_universe)
    return RunPlan(
        duration_min=duration_int,
        mode=mode,
        overhead_min=overhead,
        api_per_min=SAFE_API_PER_MIN,
        universe_target=target,
        api_call_budget=api_calls,
    )
