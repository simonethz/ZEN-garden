"""Rebuild ``run_summary.csv`` for an existing bias-sweep folder.

Recomputes every variation's row OFFLINE from its saved ``Results`` using the
corrected, reference-year-consistent net present cost (see
``run_and_visualize.record_run_summary``): instead of summing the model's saved
``net_present_cost`` -- whose discount base year is reset per rolling-horizon
step and therefore inflates the total -- ``total_cost`` is rebuilt from the
undiscounted annual ``cost_total`` and discounted positionally to the first
support year, so perfect-foresight and rolling-horizon runs are comparable.

The optimization label and subsidies (normally taken from the in-process plugin
config) are recovered here from the variation FOLDER NAME, e.g.
``bias_weight_0.5__pv_remun_DE`` -> bias weight 0.5, subsidy scenario
``pv_remun_DE``.

Usage:
    python rebuild_run_summary.py [SWEEP_DIR] [-o OUT_CSV] [--no-duals]

Defaults to the 2026-06-13 Crystal-Ball-small sweep and writes
``<SWEEP_DIR>/run_summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from zen_garden import Results

# reuse the exact helpers/ordering used by the live writer so the rebuilt CSV
# is schema-identical
from run_and_visualize import (
    DATASET_PATH,
    SUBSIDY_SCENARIOS,
    format_subsidies,
    order_run_summary_columns,
)

MODEL_NAME = DATASET_PATH.name  # inner results folder per variation ("Crystal_Ball")

# Sweep folder to rebuild the run summary for. Override here (or pass the folder
# as the first CLI argument) to point at a different bias-sweep run.
DEFAULT_SWEEP = Path(
    r"D:\Students\ssambale_jwiegner\Crystal-Ball-small\data\outputs\bias_sweep_2026-06-13_13-03-04"
)

# The subsidy descriptor for a variation is normally recovered from the live
# ``SUBSIDY_SCENARIOS`` in run_and_visualize. That list gets edited between
# experiments (it currently holds only ``no_subsidy``), so the labels used by
# older sweeps may no longer resolve. This fallback reproduces the scenarios the
# 2026-06-13 sweep ran with, matching the descriptors in its original CSV.
FALLBACK_SUBSIDY_SCENARIOS: list[tuple[str, list[dict]]] = [
    ("no_subsidy", []),
    ("pv_remun_DE", [
        {"technology": "photovoltaics", "node": "DE", "type": "remuneration",
         "amount": 0.11, "carrier": "electricity"},
    ]),
    ("hp_capex_DE", [
        {"technology": "heat_pump", "node": "DE", "type": "capex", "amount": 350},
    ]),
    ("gasboiler_co2_DE", [
        {"technology": "natural_gas_boiler", "node": "DE", "type": "variable_opex",
         "amount": 0.00710412},
    ]),
    ("gasturbine_capmarket_IT", [
        {"technology": "natural_gas_turbine", "node": "IT", "type": "fixed_opex",
         "amount": 70},
    ]),
]

# live scenarios win; fallback fills in labels the live list no longer defines
_SUBSIDY_MAP = {**dict(FALLBACK_SUBSIDY_SCENARIOS), **dict(SUBSIDY_SCENARIOS)}


def parse_label(label: str) -> tuple[str, str]:
    """Recover (optimization, subsidies-descriptor) from a variation folder name.

    Mirrors ``create_variation_configs``' labelling: ``no_bias`` /
    ``bias_weight_<w>`` optionally suffixed with ``__<subsidy_label>``.
    """
    if "__" in label:
        bias_part, sub_label = label.split("__", 1)
    else:
        bias_part, sub_label = label, None

    if bias_part == "no_bias":
        optimization = "total_cost"
    elif bias_part.startswith("bias_weight_"):
        weight = bias_part[len("bias_weight_"):]
        optimization = f"profitability_bias (bias_weight={weight})"
    else:
        optimization = bias_part  # unexpected label; keep verbatim

    subsidies = _SUBSIDY_MAP.get(sub_label, []) if sub_label else []
    return optimization, format_subsidies(subsidies)


def build_row(results_path: Path, label: str, with_duals: bool = True) -> dict:
    """Build one ``run_summary`` row from a variation's saved results.

    Replicates ``run_and_visualize.record_run_summary`` (corrected NPC, per-year
    discounted cost, system parameters, capacity additions, shadow prices) but
    sources the optimization/subsidies labels from ``label`` instead of the
    in-process plugin config.
    """
    results = Results(path=str(results_path))
    row: dict[str, float | str] = {}

    # the original in-process timestamp is lost; use the results folder's
    # modification time as a chronological proxy so rows stay ordered
    from datetime import datetime

    row["run_timestamp"] = datetime.fromtimestamp(
        results_path.stat().st_mtime
    ).isoformat(timespec="seconds")

    optimization, subsidies = parse_label(label)
    row["optimization"] = optimization
    row["subsidies"] = subsidies

    # --- system parameters ---
    scenario = next(iter(results.solution_loader.scenarios.values()))
    system = scenario.system
    optimized_years = int(getattr(system, "optimized_years", 0))
    interval = int(getattr(system, "interval_between_years", 1))
    row["optimized_years"] = optimized_years
    row["interval_between_years"] = interval
    row["aggregated_time_steps_per_year"] = int(
        getattr(system, "aggregated_time_steps_per_year", 0)
    )
    use_rolling = bool(getattr(system, "use_rolling_horizon", False))
    row["foresight_mode"] = (
        "rolling_horizon" if use_rolling else "perfect_foresight"
    )
    row["years_in_rolling_horizon"] = (
        int(getattr(system, "years_in_rolling_horizon", 0)) if use_rolling else ""
    )

    # --- total cost: NPC discounted to the reference (first support) year ---
    try:
        dr_component = scenario.get_component("discount_rate")
        r = float(
            results.solution_loader.get_component_data(
                scenario, dr_component
            ).squeeze()
        )
    except Exception:
        r = float("nan")

    cost = results.get_total("cost_total")
    if isinstance(cost, pd.DataFrame):
        cost = cost.iloc[0] if cost.shape[0] >= 1 else pd.Series(dtype=float)
    cost = pd.Series(cost)

    total_cost = float("nan")
    if len(cost) and pd.notna(r):
        n = len(cost)
        total_cost = 0.0
        for pos in range(n):
            dy = 1 if pos == n - 1 else interval
            factor = sum(
                (1.0 / (1.0 + r)) ** (interval * pos + i) for i in range(dy)
            )
            year_label = cost.index[pos]
            cost_disc_y = float(cost.iloc[pos]) * factor
            row[f"cost_disc|year_{year_label}"] = cost_disc_y
            total_cost += cost_disc_y
    row["total_cost"] = total_cost

    # --- total carbon emissions: cumulative emissions in the last optimized year ---
    total_carbon_emissions = float("nan")
    try:
        emissions = results.get_total("carbon_emissions_cumulative")
        if isinstance(emissions, pd.DataFrame):
            emissions = (
                emissions.iloc[0] if emissions.shape[0] >= 1
                else pd.Series(dtype=float)
            )
        emissions = pd.Series(emissions)
        if len(emissions):
            total_carbon_emissions = float(emissions.iloc[-1])
    except Exception as exc:
        print(f"    Warning: no total carbon emissions: {exc}")
    row["total_carbon_emissions"] = total_carbon_emissions

    # --- capacity additions per (tech, node, year) ---
    cap = results.get_df("capacity_addition")
    if isinstance(cap, dict):
        cap = next(iter(cap.values()))
    if cap is not None and len(cap) > 0:
        s = cap if isinstance(cap, pd.Series) else cap.iloc[:, 0]
        names = list(s.index.names)

        def _lvl(cands):
            for c in cands:
                if c in names:
                    return c
            return None

        tech_lvl = _lvl(["technology", "set_technologies"])
        node_lvl = _lvl(["location", "node", "set_location", "set_nodes"])
        year_lvl = _lvl(["year", "set_time_steps_yearly", "time_operation"])
        cap_lvl = _lvl(["capacity_type", "set_capacity_types"])
        if cap_lvl is not None:
            try:
                s = s.xs("power", level=cap_lvl)
            except KeyError:
                s = s.groupby(level=[l for l in names if l != cap_lvl]).sum()

        if tech_lvl and node_lvl and year_lvl:
            for (tech, node, year), val in s.groupby(
                level=[tech_lvl, node_lvl, year_lvl]
            ).sum().items():
                row[f"cap|{tech}|{node}|{year}"] = float(val)

    # --- shadow prices: mean over the full-resolution time series, per year ---
    if with_duals:
        for y in range(optimized_years):
            try:
                duals_y = results.get_dual(
                    "constraint_nodal_energy_balance", year=y
                )
            except Exception as exc:
                print(f"    Warning: no shadow prices for year {y}: {exc}")
                continue
            if duals_y is None or len(duals_y) == 0:
                continue
            mean_sp = duals_y if isinstance(duals_y, pd.Series) else duals_y.mean(axis=1)
            for key, val in mean_sp.items():
                key_tuple = key if isinstance(key, tuple) else (key,)
                parts = "|".join(str(k) for k in key_tuple)
                row[f"sp|{parts}|year_{y}"] = float(val)

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_dir", nargs="?", default=str(DEFAULT_SWEEP),
        help="bias-sweep folder containing one subfolder per variation",
    )
    parser.add_argument(
        "-o", "--out", default=None,
        help="output CSV path (default: <sweep_dir>/run_summary.csv)",
    )
    parser.add_argument(
        "--no-duals", action="store_true",
        help="skip shadow-price (dual) extraction for a faster rebuild",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep folder not found: {sweep_dir}")
    out_path = Path(args.out) if args.out else sweep_dir / "run_summary.csv"

    # each variation = a subfolder that holds the <MODEL_NAME> results folder,
    # sorted for a stable row order
    variations = sorted(
        p for p in sweep_dir.iterdir() if (p / MODEL_NAME).is_dir()
    )
    if not variations:
        raise RuntimeError(
            f"No variation subfolders with a '{MODEL_NAME}' results folder under "
            f"{sweep_dir}"
        )

    rows: list[dict] = []
    for i, var in enumerate(variations, 1):
        label = var.name
        print(f"[{i}/{len(variations)}] {label}")
        try:
            rows.append(
                build_row(var / MODEL_NAME, label, with_duals=not args.no_duals)
            )
        except Exception as exc:
            print(f"    FAILED: {exc}")

    if not rows:
        raise RuntimeError("No rows could be built; nothing written.")

    df = pd.DataFrame(rows)
    df = df.reindex(columns=order_run_summary_columns(list(df.columns)))
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nWrote {len(rows)} rows x {df.shape[1]} columns to {out_path}")


if __name__ == "__main__":
    main()
