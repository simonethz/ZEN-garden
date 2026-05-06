"""Run ZEN-garden on the Crystal-Ball-small dataset and print yearly average duals."""

from pathlib import Path

import pandas as pd

from zen_garden import Results, run

# Candidate dataset roots, tried in order. The first existing one wins, so the
# script works on both the local and the remote machine without manual edits.
DATASET_ROOT_CANDIDATES = (
    Path("D:/Students/ssambale_jwiegner/Crystal-Ball/data"),  # remote
    Path.home() / "C:\Crystal-Ball-small\data",        # local — adjust as needed
)


def get_dataset_root() -> Path:
    """Return the first existing dataset root from ``DATASET_ROOT_CANDIDATES``.

    Raises:
        FileNotFoundError: if none of the candidate paths exists on this machine.
    """
    for candidate in DATASET_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    tried = "\n  ".join(str(p) for p in DATASET_ROOT_CANDIDATES)
    raise FileNotFoundError(
        f"None of the configured dataset roots exist on this machine:\n  {tried}"
    )


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
    return yearly_average


if __name__ == "__main__":
    run_and_print_yearly_average_duals()