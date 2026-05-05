"""Helpers for extracting investor-relevant signals from a solved optimization."""

import logging

import numpy as np
import pandas as pd



def _normalize_interval(optimization_setup):
    """interval between the optimized years for normalizing the dual variables ``.
    """
    system = optimization_setup.system
    interval = system.interval_between_years
    years = list(range(0, system.optimized_years))

    factor = pd.Series(index=years, dtype=float)
    for year in years:
        factor[year] = 1 if year == years[-1] else interval
    return factor


def extract_shadow_prices_full_ts(optimization_setup):
    """Return the full (disaggregated) time series of nodal energy balance duals.

    Args:
        optimization_setup: A solved ``OptimizationSetup`` whose solver was run
            with ``save_duals = True``.


    Returns:
        pandas.DataFrame indexed by ``(set_carriers, set_nodes)`` with one
        column per base time step holding the per-hour shadow price, or
        ``None`` if duals are unavailable.
    """
    model = optimization_setup.model
    constraint ="constraint_nodal_energy_balance"
    if constraint not in model.constraints:
        logging.warning(
            f"Constraint '{constraint}' not found in the "
            "model. Cannot extract nodal energy balance duals."
        )
        return None

    duals = model.constraints[constraint].dual
    if duals is None:
        logging.warning(
            f"Duals for '{constraint}' are None. Make "
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
    annuity = _normalize_interval(optimization_setup)
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






