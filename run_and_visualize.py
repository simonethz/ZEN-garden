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
    * ``4`` – Crystal-Balll reduced auf Remote-Rechner

    Raises:
        ValueError: if ``DATASET`` is not one of the supported values (0, 1, 2, 3, 4).
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
    else:
        raise ValueError(
            f"Unsupported DATASET value {DATASET!r}; expected 0, 1, 2, 3 or 4."
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



def run_dataset():
    """Run ZEN-garden."""
    run(config=str(CONFIG_PATH), dataset=str(DATASET_PATH))


def visualize_capacity_additions():
    """Stacked bar chart of capacity additions per (node, year), stacked by tech.

    One bar per (node, year); years for the same node sit next to each other.
    Saved to ``<dataset_root_parent>/visualization/capacity_additions.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Results(path=str(OUTPUT_PATH))
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
    out_path = out_dir / "capacity_additions.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved capacity-addition plot to {out_path}")



def record_run_summary(csv_name: str = "run_summary.csv") -> None:
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
    """
    from datetime import datetime
    import csv

    results = Results(path=str(OUTPUT_PATH))
    row: dict[str, float | str] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

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
        duals_y = results.get_dual("constraint_nodal_energy_balance", year=y)
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


if __name__ == "__main__":
    run_dataset()
    visualize_capacity_additions()
    record_run_summary()
