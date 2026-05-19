"""Run ZEN-garden on the Crystal-Ball-small dataset and print yearly average duals."""

from pathlib import Path

import pandas as pd

from zen_garden import Results, run

DATASET = 1 #0 for Crystal-Ball-small remote, 1 for Crystal-Ball full remote, 2 for Crystal-Ball-small local


def get_dataset_root() -> Path:
    """Return the dataset root for the dataset selected via ``DATASET``.

    The ``DATASET`` constant (siehe Zeile 15) waehlt den Datensatz:

    * ``0`` – Crystal-Ball-small auf dem Remote-Rechner
    * ``1`` – Crystal-Ball (full) auf dem Remote-Rechner
    * ``2`` – Crystal-Ball-small auf dem lokalen Rechner

    Raises:
        ValueError: if ``DATASET`` is not one of the supported values (0, 1, 2).
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
    else:
        raise ValueError(
            f"Unsupported DATASET value {DATASET!r}; expected 0, 1 or 2."
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
HOURS_PER_YEAR = 8760


def run_and_print_yearly_average_duals():
    """Run ZEN-garden and print yearly averages of the nodal balance duals.

    Solves the model, loads the saved results, pulls the full disaggregated dual
    time series with ``Results.get_full_ts``, and reports — for every carrier
    and node — the mean shadow price of each year (each successive 8760-hour
    window of the time series).
    """
    run(config=str(CONFIG_PATH), dataset=str(DATASET_PATH))

    results = Results(path=str(OUTPUT_PATH))
    full_ts = results.get_full_ts(DUAL_NAME)
    if full_ts is None or len(full_ts) == 0:
        print(f"No full time series available for '{DUAL_NAME}'.")
        return None

    n_cols = full_ts.shape[1]
    n_years = n_cols // HOURS_PER_YEAR
    if n_years == 0:
        print(
            f"Time series only has {n_cols} columns, less than one year "
            f"({HOURS_PER_YEAR} h). Reporting overall mean instead."
        )
        return full_ts.mean(axis=1).rename("average_shadow_price")

    yearly_means = {
        year: full_ts.iloc[:, year * HOURS_PER_YEAR : (year + 1) * HOURS_PER_YEAR].mean(
            axis=1
        )
        for year in range(n_years)
    }
    yearly_average = pd.concat(yearly_means, axis=1)
    yearly_average.columns.name = "year"

    print(
        f"\nYearly average nodal energy balance duals "
        f"(per carrier and node, {n_years} year(s) of {HOURS_PER_YEAR} h):\n"
        f"{yearly_average}"
    )

    flow = results.get_df("flow_conversion_output")
    print(f"\nFlow conversion output (get df series):\n{flow}")
    





if __name__ == "__main__":
    run_and_print_yearly_average_duals()