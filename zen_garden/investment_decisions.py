"""Helpers for extracting investor-relevant signals from a solved optimization."""

import logging

import numpy as np
import pandas as pd

CONSTRAINT_NODAL_ENERGY_BALANCE = "constraint_nodal_energy_balance"


def _compute_annuity(optimization_setup, discount_to_first_step=True):
    """Per-year annuity factor matching ``Results._get_annuity``.

    Mirrors the postprocessing convention so in-run duals are directly
    comparable to ``Results.get_full_ts``: each year's annuity equals the
    interval-length weight (``interval_between_years``, except the final year
    which gets 1) times the discount factor back to the corresponding decision
    step.
    """
    system = optimization_setup.system
    discount_rate = float(np.asarray(optimization_setup.parameters.discount_rate).item())
    interval = system.interval_between_years
    years = list(range(0, system.optimized_years))

    # Years that start a fresh decision step. In rolling horizon this is
    # ``optimized_time_steps``; in single-shot mode it defaults to [0].
    optimized_years = getattr(optimization_setup, "optimized_time_steps", None)
    if not optimized_years:
        optimized_years = [0]

    annuity = pd.Series(index=years, dtype=float)
    for year in years:
        start_year = [y for y in optimized_years if y <= year][-1]
        interval_this_year = 1 if year == years[-1] else interval
        if discount_to_first_step:
            annuity[year] = interval_this_year * (
                (1 / (1 + discount_rate)) ** (interval * (year - start_year))
            )
        else:
            annuity[year] = sum(
                (1 / (1 + discount_rate)) ** (interval * (year - start_year) + i)
                for i in range(interval_this_year)
            )
    return annuity


def extract_shadow_prices_full_ts(optimization_setup, discount_to_first_step=True):
    """Return the full (disaggregated) time series of nodal energy balance duals.

    The shadow prices of ``constraint_nodal_energy_balance`` are extracted from
    the solved model, normalized to a per-hour value (raw dual / duration of
    the aggregated time step), divided by the per-year annuity (so the values
    match the convention of ``Results.get_full_ts``), and finally mapped from
    the aggregated operational time steps back to the underlying base time
    steps via ``energy_system.time_steps.sequence_time_steps_operation``.

    Args:
        optimization_setup: A solved ``OptimizationSetup`` whose solver was run
            with ``save_duals = True``.
        discount_to_first_step: Apply the annuity to the first year of the
            interval (``True``) or summed over the entire interval (``False``).
            Same semantics as in ``Results.get_full_ts``.

    Returns:
        pandas.DataFrame indexed by ``(set_carriers, set_nodes)`` with one
        column per base time step holding the per-hour shadow price, or
        ``None`` if duals are unavailable.
    """
    model = optimization_setup.model
    if CONSTRAINT_NODAL_ENERGY_BALANCE not in model.constraints:
        logging.warning(
            f"Constraint '{CONSTRAINT_NODAL_ENERGY_BALANCE}' not found in the "
            "model. Cannot extract nodal energy balance duals."
        )
        return None

    duals = model.constraints[CONSTRAINT_NODAL_ENERGY_BALANCE].dual
    if duals is None:
        logging.warning(
            f"Duals for '{CONSTRAINT_NODAL_ENERGY_BALANCE}' are None. Make "
            "sure `solver.save_duals` is enabled and the solver returned duals."
        )
        return None

    # raw duals indexed by (carrier, node, aggregated operation time step)
    df_duals = duals.to_dataframe(name="dual").unstack("set_time_steps_operation")
    df_duals.columns = df_duals.columns.get_level_values("set_time_steps_operation")

    time_steps = optimization_setup.energy_system.time_steps

    # divide by duration: the raw dual equals duration * per-hour price because
    # operational variables enter the objective weighted by
    # `time_steps_operation_duration`.
    durations = pd.Series(time_steps.time_steps_operation_duration).reindex(
        df_duals.columns
    )
    df_duals = df_duals.div(durations, axis=1)

    # divide by per-year annuity, matching `Results.get_full_ts` for duals.
    annuity = _compute_annuity(optimization_setup, discount_to_first_step)
    op2year = pd.Series(time_steps.time_steps_operation2year)
    annuity_per_op = op2year.reindex(df_duals.columns).map(annuity)
    df_duals = df_duals.div(annuity_per_op, axis=1)

    # map aggregated operation time steps onto the underlying base time steps
    sequence = np.asarray(time_steps.sequence_time_steps_operation)
    full_ts = df_duals.reindex(columns=sequence)
    full_ts.columns = pd.RangeIndex(len(sequence), name="base_time_step")

    return full_ts

def extract_average_shadow_prices(optimization_setup):
    """Return the average shadow price of nodal energy balance duals across time.

    The shadow prices of ``constraint_nodal_energy_balance`` are extracted from
    the solved model, normalized to a per-hour value (raw dual / duration of the
    aggregated time step), and averaged across all aggregated operational time
    steps.

    Args:
        optimization_setup: A solved ``OptimizationSetup`` whose solver was run
            with ``save_duals = True``.

    Returns:
        pandas.DataFrame indexed by ``(set_carriers, set_nodes)`` with a single
        column holding the average per-hour shadow price, or ``None`` if
        duals are unavailable.
    """
    full_ts = extract_shadow_prices_full_ts(optimization_setup)
    if full_ts is None:
        return None

    return full_ts.mean(axis=1).to_frame(name="average_shadow_price")


def print_average_shadow_prices(optimization_setup):
    """Print the average shadow prices of nodal energy balance duals.

    Args:
        optimization_setup: A solved ``OptimizationSetup`` whose solver was run
            with ``save_duals = True``.
    """
    average_prices = extract_average_shadow_prices(optimization_setup)
    if average_prices is not None:
        logging.info(f"\n--- Average Shadow Prices ---\n{average_prices}\n")
    else:
        logging.warning("No average shadow prices available to print.")



