"""Run ZEN-garden on the Crystal-Ball-small dataset and print yearly average duals."""

from pathlib import Path

import pandas as pd

from zen_garden import Results, run

# set dataset root by selecting one of the predefined paths in `get_dataset_root()`; adjust the paths in that function as needed to run on your machine
DATASET = 0

#bias weights to run in one program start; ``None`` = total_cost
#BIAS_WEIGHTS: list[float | None] = [None, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50, 100]
#BIAS_WEIGHTS: list[float | None] = [None, 0.2, 0.5, 0.8, 5.0]
BIAS_WEIGHTS: list[float | None] = [None]


# Subsidy scenarios to run in one program start. 
#   - "capex"         : one-time relief in the decision year [money/GW added]
#   - "fixed_opex"    : annual lump-sum relief [money/GW added], discounted
#   - "variable_opex" : per-MWh relief [money/MWh] on the reference-carrier flow
#   - "remuneration"  : fixed feed-in price [money/MWh] for an output "carrier" (requires the extra key "carrier")
# The full sweep is the cross product BIAS_WEIGHTS x SUBSIDY_SCENARIOS. Keep the default first entry to include the no-subsidy baseline.
SUBSIDY_SCENARIOS: list[tuple[str, list[dict]]] = [
    ("no_subsidy", []),
    #("pv_remun_DE", [ {"technology": "photovoltaics", "node": "DE", "type": "remuneration", "amount": 0.11, "carrier": "electricity"}, ]),
    #("hp_capex_DE", [ {"technology": "heat_pump", "node": "DE", "type": "capex", "amount": 350}, ]),
    #("gasboiler_co2_DE", [ {"technology": "natural_gas_boiler", "node": "DE", "type": "variable_opex", "amount": 0.00710412}, ]),
    #("gasturbine_capmarket_IT", [ {"technology": "natural_gas_turbine", "node": "IT", "type": "fixed_opex", "amount": 70}, ]),
]

# bias acts only on conversion technologies with at least one of these output carriers; if empty, bias applies to all output carriers
BIAS_OUTPUT_CARRIERS: list[str] = []


def get_dataset_root() -> Path:
    """Return the dataset root for the dataset selected via ``DATASET``.

    The ``DATASET`` constant (siehe Zeile 15) waehlt den Datensatz:

    * ``0`` – Crystal-Ball-small auf dem Remote-Rechner
    * ``1`` – Crystal-Ball (full) auf dem Remote-Rechner
    * ``2`` – Crystal-Ball-small auf dem lokalen Rechner
    * ``3`` – Climate resilience auf Remote-Rechner
    * ``4`` – Crystal-Ball reduced auf Remote-Rechner
    * ``5`` – Crystal-Ball-origial auf remote Rechner
    """

    if DATASET == 0:
        # Crystal-Ball-small, remote 
        root = Path("D:/Students/ssambale_jwiegner/Crystal-Ball-small/data") 
    elif DATASET == 1:
        # Crystal-Ball (full), remote
        root = Path("D:/Students/ssambale_jwiegner/Crystal-Ball/data")
    elif DATASET == 2:
        # Crystal-Ball-small, local 
        root = Path("C:/Crystal-Ball-small/data") 
    elif DATASET == 3:
        # Climate resilience, remote
        root = Path("D:/Students/ssambale_jwiegner/Climate-resilience/data")
    elif DATASET == 4:
        # Crystal-Ball reduced, remote
        root = Path("D:/Students/ssambale_jwiegner/Crystal_Ball_reduced/data")
    elif DATASET == 5:
        # Crystal-Ball-original, remote
        root = Path("D:/Students/ssambale_jwiegner/Crystal-Ball-original/data")
    else:
        raise ValueError(
            f"Unsupported DATASET value {DATASET!r}; expected 0, 1, 2, 3, 4 or 5."
        )

    if not root.exists():
        raise FileNotFoundError(
            f"Selected dataset root does not exist on this machine:\n  {root}"
        )
    return root


DATASET_ROOT = get_dataset_root()
CONFIG_PATH = DATASET_ROOT / "config.json"
DATASET_PATH = DATASET_ROOT / "Crystal_Ball"
OUTPUT_PATH = DATASET_ROOT / "outputs" / "Crystal_Ball"
DUAL_NAME = "constraint_nodal_energy_balance"



def run_dataset(config_path: Path | None = None, folder_output: Path | None = None):
    """Run ZEN-garden.

    Args:
        config_path: config file to use; defaults to the dataset's ``config.json``.
        folder_output: output folder for this run; defaults to the location
            configured in the config file (``outputs``).
    """
    run(
        config=str(config_path or CONFIG_PATH),
        dataset=str(DATASET_PATH),
        folder_output=str(folder_output) if folder_output is not None else None,
    )


# capacity additions diagramms
def visualize_capacity_additions(
    output_path: Path | None = None,
    plot_name: str = "capacity_additions.png",
):
    """Stacked bar chart of capacity additions per (node, year), stacked by tech.

    One bar per (node, year); years for the same node sit next to each other.
    Saved to ``<dataset_root_parent>/visualization/<plot_name>``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Results(path=str(output_path or OUTPUT_PATH))
    df = results.get_df("capacity_addition")
    if df is None or len(df) == 0:
        print("No capacity_addition data available.")
        return

    if isinstance(df, dict):
        df = next(iter(df.values()))
    s = df if isinstance(df, pd.Series) else df.iloc[:, 0]

    def _find_level(names, candidates):
        for c in candidates:
            if c in names:
                return c
        raise KeyError(f"None of {candidates} in index levels {list(names)}")

    names = s.index.names
    tech_lvl = _find_level(names, ["technology", "set_technologies"])
    node_lvl = _find_level(
        names, ["location", "node", "set_location", "set_nodes"]
    )
    year_lvl = _find_level(
        names, ["year", "set_time_steps_yearly", "time_operation"]
    )
    cap_lvl_candidates = ["capacity_type", "set_capacity_types"]
    cap_lvl = next((c for c in cap_lvl_candidates if c in names), None)
    if cap_lvl is not None:
        s = s.xs("power", level=cap_lvl)

    pivot = s.unstack(tech_lvl).fillna(0.0)
    pivot = pivot.reorder_levels([node_lvl, year_lvl]).sort_index()

    # drop techs that never get added
    pivot = pivot.loc[:, (pivot != 0).any(axis=0)]
    if pivot.empty:
        print("No non-zero capacity additions to plot.")
        return

    labels = [f"{node}\n{year}" for node, year in pivot.index]
    x = list(range(len(labels)))
    fig_width = max(8, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bottom = pd.Series(0.0, index=pivot.index)
    cmap = plt.colormaps["tab20"]
    totals = pivot.sum(axis=1).values
    label_threshold = max(totals.max() * 0.03, 1e-9) if len(totals) else 0.0
    for i, tech in enumerate(pivot.columns):
        vals = pivot[tech].values
        ax.bar(x, vals, bottom=bottom.values, label=str(tech),
               color=cmap(i % cmap.N), edgecolor="white", linewidth=0.3)
        for xi, v, b in zip(x, vals, bottom.values):
            if v > label_threshold:
                ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        bottom = bottom + pivot[tech]

    for xi, total in zip(x, totals):
        if total > 0:
            ax.text(xi, total, f"{total:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    # visual separators between nodes
    nodes_in_order = [n for n, _ in pivot.index]
    for i in range(1, len(nodes_in_order)):
        if nodes_in_order[i] != nodes_in_order[i - 1]:
            ax.axvline(i - 0.5, color="black", linewidth=0.5, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Capacity addition [GW]")
    ax.set_title("Capacity additions per node and year")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out_dir = DATASET_ROOT.parent / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / plot_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved capacity-addition plot to {out_path}")



def format_subsidies(subsidies) -> str:
    """Compact, CSV-friendly descriptor of a list of subsidy entries.

    Each entry (plugin format ``{technology, node, type, amount, carrier?}``) is
    rendered as ``<tech>@<node>:<type>=<amount>[/<carrier>]``; entries are joined
    with ``;``. Returns ``"none"`` for an empty/missing list.
    """
    if not subsidies:
        return "none"
    parts = []
    for s in subsidies:
        try:
            piece = f"{s['technology']}@{s['node']}:{s['type']}={s['amount']:g}"
        except (KeyError, TypeError, ValueError):
            parts.append(str(s))
            continue
        if s.get("carrier"):
            piece += f"/{s['carrier']}"
        parts.append(piece)
    return ";".join(parts)


def order_run_summary_columns(cols: list[str]) -> list[str]:
    """Left-to-right layout for ``run_summary.csv``.    """

    preferred = [
        "run_timestamp", "optimization", "subsidies", "total_cost",
        "total_carbon_emissions",
        "optimized_years", "interval_between_years",
        "aggregated_time_steps_per_year", "foresight_mode",
        "years_in_rolling_horizon",
    ]
    cap_cols = [c for c in cols if c.startswith("cap|")]
    sp_cols = [c for c in cols if c.startswith("sp|")]
    cost_year = sorted(
        (c for c in cols if c.startswith("cost_disc|year_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    front = [c for c in preferred if c in cols]
    used = set(front) | set(cap_cols) | set(sp_cols) | set(cost_year)
    other = [c for c in cols if c not in used]
    return front + cost_year + other + cap_cols + sp_cols



def record_run_summary(
    csv_name: str = "run_summary.csv",
    output_path: Path | None = None,
) -> None:
    """Append a one-line snapshot of the current run to a shared CSV.

    Columns (header only in row 1):
      - run_timestamp
      - total_cost: sum of `net_present_cost` over all optimized years
      - cap|<tech>|<node>|<year>: capacity addition per (tech, node, year)
      - sp|<carrier>|<node>|<year>: mean shadow price of the nodal energy
        balance over the full-resolution time series (analogous to
        `extract_normalized_dual` -> averaged across time)

    """
    from datetime import datetime
    import csv

    results = Results(path=str(output_path or OUTPUT_PATH))
    row: dict[str, float | str] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # --- optimization mode + bias value (from the investment_decisions plugin config, populated in-process by register_plugins during the run) ---
    try:
        from zen_garden.plugins.investment_decisions.plugin import (
            config as plugin_config,
        )
    except Exception:
        plugin_config = {}
    bias_enabled = bool(plugin_config.get("profitability_bias_enabled", False))
    bias_weight = plugin_config.get("bias_weight", 0.5)
    if bias_enabled:
        label = f"profitability_bias (bias_weight={bias_weight}"
        bias_carriers = plugin_config.get("bias_output_carriers")
        if bias_carriers:
            if isinstance(bias_carriers, str):
                bias_carriers = [bias_carriers]
            label += f", output_carriers={'/'.join(bias_carriers)}"
        label += ")"
        row["optimization"] = label
    else:
        row["optimization"] = "total_cost"

    # --- subsidies active this run (from the same in-process plugin config) ---
    subsidies = plugin_config.get("subsidies", []) or []
    row["subsidies"] = format_subsidies(subsidies)

    # --- system parameters of this run ---
    try:
        scenario = next(iter(results.solution_loader.scenarios.values()))
        system = scenario.system
    except Exception:
        scenario = None
        system = None

    optimized_years = int(getattr(system, "optimized_years", 0)) if system else 0
    interval = int(getattr(system, "interval_between_years", 1)) if system else 1
    if system is not None:
        row["optimized_years"] = optimized_years
        row["interval_between_years"] = interval
        row["aggregated_time_steps_per_year"] = int(
            getattr(system, "aggregated_time_steps_per_year", 0)
        )
        use_rolling = bool(getattr(system, "use_rolling_horizon", False))
        row["foresight_mode"] = (
            "rolling_horizon" if use_rolling else "perfect_foresight"
        )
        # only meaningful in rolling horizon; left blank for perfect foresight
        row["years_in_rolling_horizon"] = (
            int(getattr(system, "years_in_rolling_horizon", 0)) if use_rolling else ""
        )

    # --- total cost: net present cost discounted to the reference year to enable comparison of rolling horizon and perfect foresight ---
    r = float("nan")
    if scenario is not None:
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
        print(f"  Warning: could not extract total carbon emissions: {exc}")
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
    n_years = optimized_years

    for y in range(n_years):
        try:
            duals_y = results.get_dual("constraint_nodal_energy_balance", year=y)
        except Exception as exc:
            print(f"  Warning: could not extract shadow prices for year {y}: {exc}")
            continue
        if duals_y is None or len(duals_y) == 0:
            continue
        if isinstance(duals_y, pd.Series):
            mean_sp = duals_y
        else:
            mean_sp = duals_y.mean(axis=1)

        for key, val in mean_sp.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            parts = "|".join(str(k) for k in key_tuple)
            row[f"sp|{parts}|year_{y}"] = float(val)

    # --- append to CSV (union of columns across runs) ---
    out_dir = DATASET_ROOT.parent / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / csv_name

    if out_path.exists():
        existing = pd.read_csv(out_path)
        all_cols = list(existing.columns)
        for c in row.keys():
            if c not in all_cols:
                all_cols.append(c)
        all_cols = order_run_summary_columns(all_cols)
        new_row_df = pd.DataFrame([row]).reindex(columns=all_cols)
        if list(existing.columns) != all_cols:
            existing = existing.reindex(columns=all_cols)
            combined = pd.concat([existing, new_row_df], ignore_index=True)
            combined.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        else:
            new_row_df.to_csv(
                out_path, mode="a", header=False, index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
    else:
        cols = order_run_summary_columns(list(row.keys()))
        pd.DataFrame([row]).reindex(columns=cols).to_csv(
            out_path, index=False, quoting=csv.QUOTE_MINIMAL,
        )
    print(f"Appended run summary to {out_path}")




def _latest_profitability_csv() -> Path | None:
    """Return the most recent ``profitability_components.csv`` of this dataset.

    The investment_decisions plugin writes one CSV per program run into a
    timestamped subfolder of ``<dataset_root>/visualization``. This picks the
    newest such file (by modification time), or ``None`` if none exists.
    """
    viz = DATASET_ROOT.parent / "visualization"
    candidates = list(viz.glob("*/profitability_components.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def visualize_profitability_over_years(csv_path: str | Path | None = None) -> None:
    """For each output carrier, plot the profitability development over decision years.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if csv_path is None:
        csv_path = _latest_profitability_csv()
    if csv_path is None:
        print("No profitability_components.csv found; skipping plots.")
        return
    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"{csv_path} is empty; skipping plots.")
        return

    # one technology may produce several output carriers -> split and explode so each row is attributed to every carrier it contributes to.
    df["output_carrier"] = df["output_carrier"].fillna("").astype(str)
    df = df.assign(output_carrier=df["output_carrier"].str.split(", ")).explode(
        "output_carrier"
    )
    df = df[df["output_carrier"].str.strip() != ""]
    if df.empty:
        print(f"{csv_path} has no output carriers; skipping plots.")
        return

    out_dir = csv_path.parent
    cmap = plt.colormaps["tab10"]

    for carrier, carrier_df in df.groupby("output_carrier"):
        fig, ax = plt.subplots(figsize=(9, 5))

        plotted = 0
        for i, ((tech, node), pair_df) in enumerate(
            carrier_df.groupby(["set_conversion_technologies", "set_nodes"])
        ):
            pair_df = pair_df.sort_values("decision_year")
            ax.plot(
                pair_df["decision_year"],
                pair_df["profitability"],
                marker="o",
                color=cmap(i % cmap.N),
                label=f"{tech} / {node}",
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlabel("Decision year")
        ax.set_ylabel("Net discounted profitability [model money units]")
        ax.set_title(f"Profitability development – output carrier '{carrier}'")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.grid(axis="both", linestyle=":", alpha=0.5)

        years = sorted(carrier_df["decision_year"].unique())
        ax.set_xticks(years)

        out_path = out_dir / f"profitability_over_years_{carrier}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved profitability-over-years plot to {out_path}")





def create_variation_configs(
    bias_weights, subsidy_scenarios=None
) -> list[tuple[str, Path]]:
    """Write one config file per (bias weight x subsidy scenario) combination.

    Args:
        bias_weights: iterable of weights, e.g. ``[None, 0.25, 0.5, 1.0]``.
        subsidy_scenarios: iterable of ``(sub_label, subsidies)`` pairs, where
            ``subsidies`` is a list of subsidy entries (possibly empty).
            Defaults to a single no-subsidy scenario, so the behaviour is
            identical to a pure bias-weight sweep. The full sweep is the cross
            product ``bias_weights x subsidy_scenarios``.

    Returns:
        list of ``(label, config_path)`` tuples in input order.
    """
    import copy
    import json

    if subsidy_scenarios is None:
        subsidy_scenarios = [("no_subsidy", [])]

    with open(CONFIG_PATH) as f:
        base_config = json.load(f)

    variations: list[tuple[str, Path]] = []
    for weight in bias_weights:
        bias_label = "no_bias" if weight is None else f"bias_weight_{weight}"
        for sub_label, subsidies in subsidy_scenarios:
            #subsidy label
            label = (
                bias_label
                if sub_label in (None, "", "no_subsidy") or not subsidies
                else f"{bias_label}__{sub_label}"
            )
            cfg = copy.deepcopy(base_config)
            plugin_cfg = cfg.setdefault("plugins", {}).setdefault(
                "investment_decisions", {}
            )
            if weight is None:
                plugin_cfg["profitability_bias_enabled"] = False
            else:
                plugin_cfg["profitability_bias_enabled"] = True
                plugin_cfg["bias_weight"] = weight
                plugin_cfg["bias_output_carriers"] = list(BIAS_OUTPUT_CARRIERS)

            plugin_cfg["subsidies"] = copy.deepcopy(list(subsidies))
           
            if subsidies and sub_label not in (None, "", "no_subsidy"):
                plugin_cfg["subsidy_label"] = sub_label

            config_path = DATASET_ROOT / f"config_{label}.json"
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=4)
            variations.append((label, config_path))
            print(f"Created variation config: {config_path}")
    return variations





if __name__ == "__main__":
    from datetime import datetime

    from zen_garden.plugins.investment_decisions import investment_decisions

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_root = DATASET_ROOT / "outputs" / f"bias_sweep_{time_str}"

    for label, config_file in create_variation_configs(BIAS_WEIGHTS, SUBSIDY_SCENARIOS):
        print(f"\n================ Variation: {label} ================\n")
        result_folder = sweep_root / label
        result_folder.mkdir(parents=True, exist_ok=True)

        investment_decisions._RUN_TIMESTAMP = None

        run_dataset(config_file, result_folder)

        # results of this variation live in <result_folder>/<model_name>
        output_path = result_folder / DATASET_PATH.name
        visualize_capacity_additions(
            output_path, plot_name=f"capacity_additions_{label}.png"
        )
        record_run_summary(output_path=output_path)
        visualize_profitability_over_years()
