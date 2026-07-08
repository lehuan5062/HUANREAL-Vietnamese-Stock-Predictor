"""Compare rebound_sim_include_held vs rebound_sim_exclude_held (limit_next_day).

Both buy daily. Only difference: exclude_held removes currently-held tickers
from the candidate pool (buys the best UNHELD pick), while include_held can
re-buy a name already owned.

Outputs: reports/sim_outputs/include_vs_exclude_held.json
"""
import json
import os
from scripts import rebound_sim_include_held as inc
from scripts import rebound_sim_exclude_held as exc

print("Building data...", flush=True)
data = inc._build_data()

print("Running include_held...", flush=True)
include_result = inc.simulate(data=data)

print("Running exclude_held...", flush=True)
exclude_result = exc.simulate(data=data)

comparison = {
    "include_held": include_result,
    "exclude_held": exclude_result,
}

os.makedirs("reports/sim_outputs", exist_ok=True)
dest = "reports/sim_outputs/include_vs_exclude_held.json"
with open(dest, "w") as f:
    json.dump(comparison, f, indent=2, default=str)

print(f"WROTE {dest}", flush=True)
