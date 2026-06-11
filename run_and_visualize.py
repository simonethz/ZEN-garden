"""Run ZEN-garden on the Crystal-Ball-small dataset and print yearly average duals."""

from pathlib import Path

import pandas as pd

from zen_garden import Results, run

DATASET = 0 #0 for Crystal-Ball-small remote, 1 for Crystal-Ball full remote, 2 for Crystal-Ball-small local, 3 for Climate resilience remote, 4 for Crystal-Ball reduced remote


def get_dataset_root() -> Path:
    """Return the dataset root for the dataset selected via ``DATASET``.

    The ``DATASET`` constant (siehe Zeile 15) waehlt den Datensatz:

    * ``0`` – Crystal-Ball-small auf dem Remote-Rechner
    * ``1`` – Crystal-Ball (full) auf dem Remote-Rechner
    * ``2`` – Crystal-Ball-small auf dem lokalen Rechner
    * ``3`` – Climate resilience auf Remote-Rechner
    * ``4`` – Crystal-Ball reduced auf Remote-Rechner
    * ``5`` – Crystal-Ball-origial auf remote Rechner

    Raises:
        ValueError: if ``DATASET`` is not one of the supported values (0, 1, 2, 3, 4, 5).
        FileNotFoundError: if the selected dataset root does not exist.
    """
    if DATASET == 0:
        # Crystal-Ball-small, remote -- Pfad ggf. anpassen
        root = Path("D:/Students/ssambale_jwiegner/Crystal-Ball-small/data") 
    elif DATASET == 1:
        # Crystal-Ball (full), remote
        root = Path("D:/Students/ssambale_jwiegner/Crystal-Ball/data")
    elif DATASET == 2:
        # Crystal-Ball-small, local -- Pfad ggf. anpassen
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


def visualize_capacity_additions(
    output_path: Path | None = None,
    plot_name: str = "capacity_additions.png",
):
    """Stacked bar chart of capacity additions per (node, year), stacked by tech.

    One bar per (node, year); years for the same node sit next to each other.
    Saved to ``<dataset_root_parent>/visualization/<plot_name>``.

    Args:
        output_path: results folder to read; defaults to ``OUTPUT_PATH``.
        plot_name: file name of the saved chart; override per variation so
            sweep runs do not overwrite each other.
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

    Location: same as `visualize_capacity_additions` (DATASET_ROOT.parent /
    "visualization"), so columns across runs of the same dataset are aligned.

    Args:
        csv_name: file name of the shared summary CSV.
        output_path: results folder to read; defaults to ``OUTPUT_PATH``.
    """
    from datetime import datetime
    import csv

    results = Results(path=str(output_path or OUTPUT_PATH))
    row: dict[str, float | str] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # --- optimization mode + bias value (from the investment_decisions plugin
    # config, populated in-process by register_plugins during the run) ---
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

    # --- total cost ---
    npc = results.get_total("net_present_cost")
    if isinstance(npc, pd.DataFrame):
        npc = npc.iloc[:, 0] if npc.shape[1] else pd.Series(dtype=float)
    total_cost = float(pd.Series(npc).sum()) if npc is not None else float("nan")
    row["total_cost"] = total_cost

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

    # --- shadow prices: mean over the full-resolution time series, PER YEAR ---
    # `get_dual` with `year=y` returns the dual columns for the base time steps
    # of optimized year `y` only. We take the row-wise mean across those columns
    # so each (carrier, node, year) gets one number, then emit a column
    # `sp|<carrier>|<node>|year_<y>`.
    try:
        first_scenario = next(iter(results.solution_loader.scenarios.values()))
        n_years = int(first_scenario.system.optimized_years)
    except Exception:
        n_years = 0

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
        # keep the optimization column at the second position across schema changes
        if "optimization" in all_cols:
            all_cols.insert(1, all_cols.pop(all_cols.index("optimization")))
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
        pd.DataFrame([row]).to_csv(
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

    Reads ``profitability_components.csv`` (written per optimization step by the
    investment_decisions plugin) and draws, per output carrier, one line per
    (technology, node) pair showing the net discounted profitability across the
    successive ``decision_year`` values of a rolling-horizon run.

    A technology with several output carriers appears in several carrier charts
    (its ``output_carrier`` cell holds the carriers joined with ``", "`` and is
    split here). Charts are saved next to the CSV as
    ``profitability_over_years_<carrier>.png``.

    Args:
        csv_path: path to the CSV. Defaults to the newest
            ``profitability_components.csv`` under
            ``<dataset_root>/visualization``.
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

    # one technology may produce several output carriers -> split and explode so
    # each row is attributed to every carrier it contributes to.
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
        # integer ticks on the decision-year axis
        years = sorted(carrier_df["decision_year"].unique())
        ax.set_xticks(years)

        out_path = out_dir / f"profitability_over_years_{carrier}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved profitability-over-years plot to {out_path}")


# restrict the profitability bias (and thereby the ratio_min normalization) to
# technologies producing at least one of these carriers; note that the
# district-heating techs (*_DH) output "district_heat", not "heat"
BIAS_OUTPUT_CARRIERS: list[str] = []


def create_variation_configs(bias_weights) -> list[tuple[str, Path]]:
    """Write one config file per bias weight next to the base ``config.json``.

    Copies the dataset's ``config.json`` and overwrites the
    ``investment_decisions`` plugin entry: a ``None`` weight creates a baseline
    config with the profitability bias disabled, a numeric weight enables the
    bias with that ``bias_weight`` and restricts it to the output carriers in
    ``BIAS_OUTPUT_CARRIERS``. The variants are written as
    ``config_<label>.json`` into ``DATASET_ROOT`` (same directory as the base
    config, so relative paths inside the config resolve identically).

    Args:
        bias_weights: iterable of weights, e.g. ``[None, 0.25, 0.5, 1.0]``.

    Returns:
        list of ``(label, config_path)`` tuples in input order.
    """
    import copy
    import json

    with open(CONFIG_PATH) as f:
        base_config = json.load(f)

    variations: list[tuple[str, Path]] = []
    for weight in bias_weights:
        label = "no_bias" if weight is None else f"bias_weight_{weight}"
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

        config_path = DATASET_ROOT / f"config_{label}.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)
        variations.append((label, config_path))
        print(f"Created variation config: {config_path}")
    return variations


# bias weights to run in one program start; ``None`` = baseline without bias
#BIAS_WEIGHTS: list[float | None] = [None, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0, 50, 100] 
#BIAS_WEIGHTS: list[float | None] = [None, 0.1, 0.3, 0.5]
BIAS_WEIGHTS: list[float | None] = [0.3]



if __name__ == "__main__":
    from datetime import datetime

    from zen_garden.plugins.investment_decisions import investment_decisions

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_root = DATASET_ROOT / "outputs" / f"bias_sweep_{time_str}"

    for label, config_file in create_variation_configs(BIAS_WEIGHTS):
        print(f"\n================ Variation: {label} ================\n")
        result_folder = sweep_root / label
        # pre-create the nested folder: ZEN-garden's setup_output_folder uses
        # plain os.mkdir, which cannot create more than one level at once
        result_folder.mkdir(parents=True, exist_ok=True)

        # new timestamped visualization folder per variation, so the
        # profitability CSVs/plots of the variations do not mix
        investment_decisions._RUN_TIMESTAMP = None

        run_dataset(config_file, result_folder)

        # results of this variation live in <result_folder>/<model_name>
        output_path = result_folder / DATASET_PATH.name
        visualize_capacity_additions(
            output_path, plot_name=f"capacity_additions_{label}.png"
        )
        record_run_summary(output_path=output_path)
        visualize_profitability_over_years()
