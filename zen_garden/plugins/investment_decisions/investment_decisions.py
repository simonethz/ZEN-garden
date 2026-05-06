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

    logging.info(
        "\n--- Normalized shadow prices per aggregated operational time step ---\n"
        f"{df_duals}\n"
    )

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

# revenue calculation


def get_lifetime(optimization_setup) -> pd.Series:
    """Lifetime per conversion technology, in calendar years.

    Returns:
        pandas.Series indexed by ``set_technologies`` (restricted to conversion
        technologies) holding the lifetime parameter from the model.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    # parameters.lifetime is an xarray.DataArray indexed by set_technologies;
    # to_series() preserves that index, unlike pd.Series(xarr, ...) which
    # falls back to a positional integer index.
    lifetime = optimization_setup.parameters.lifetime.to_series()
    lifetime.name = "lifetime"
    return lifetime.reindex(techs)


def get_discount_rate(optimization_setup) -> pd.Series:
    """Discount rate per (conversion technology, node).

    The model currently has only a system-wide scalar ``discount_rate``, so
    every (tech, node) pair receives the same value. The (tech, node) shape
    is kept so callers don't need to change once a tech-specific override is
    added.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    rate = float(np.asarray(optimization_setup.parameters.discount_rate).item())
    idx = pd.MultiIndex.from_product(
        [techs, nodes], names=["set_technologies", "set_nodes"]
    )
    return pd.Series(rate, index=idx, name="discount_rate")


def get_market_value(optimization_setup, default: float = 0.35) -> pd.Series:
    """Market value factor per (conversion technology, node).

    Hardcoded to ``default`` (0.35) for now — replace this stub with a proper
    lookup once tech-/node-specific market values are available.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    idx = pd.MultiIndex.from_product(
        [techs, nodes], names=["set_technologies", "set_nodes"]
    )
    return pd.Series(default, index=idx, name="market_value")


def get_yearly_production(optimization_setup, year: int | None = None) -> pd.Series:
    """Specific yearly production per unit installed capacity.

    For each conversion technology, output carrier and node, returns the
    yearly output (sum over operational time steps of
    ``flow_conversion_output * time_steps_operation_duration``) divided by
    the installed capacity in that year. The result is a per-GW-installed
    full-load equivalent that can be multiplied by an arbitrary capacity
    addition to estimate annual production.

    Args:
        optimization_setup: A solved ``OptimizationSetup``.
        year: Year index in ``set_time_steps_yearly``. Defaults to the most
            recent year solved in the optimization (the "previous year"
            relative to any new investment).

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_output_carriers``, ``set_nodes``).
        ``NaN`` where capacity is zero.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    if year is None:
        year = max(sets["set_time_steps_yearly"])

    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_steps_in_year = op2year[op2year == year].index

    flow = (
        optimization_setup.model.solution["flow_conversion_output"]
        .to_series()
        .dropna()
    )
    op_level = "set_time_steps_operation"
    flow = flow[flow.index.get_level_values(op_level).isin(op_steps_in_year)]
    flow = flow.mul(flow.index.get_level_values(op_level).map(durations))
    flow = flow[
        flow.index.get_level_values("set_conversion_technologies").isin(techs)
    ]
    yearly_prod = flow.groupby(
        level=["set_conversion_technologies", "set_output_carriers", "set_nodes"]
    ).sum()
    yearly_prod.index.set_names(
        ["set_technologies", "set_output_carriers", "set_nodes"], inplace=True
    )

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sel(set_time_steps_yearly=year)
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[
        capacity.index.get_level_values("set_technologies").isin(techs)
    ]
    capacity.index.set_names(["set_technologies", "set_nodes"], inplace=True)

    cap_aligned = capacity.reindex(
        yearly_prod.index.droplevel("set_output_carriers")
    )
    cap_aligned.index = yearly_prod.index
    specific = yearly_prod.div(cap_aligned).replace([np.inf, -np.inf], np.nan)
    specific.name = "specific_yearly_production"
    return specific


def get_shadow_price(optimization_setup) -> pd.DataFrame | None:
    """Yearly average shadow price for the output carriers of all conversion techs.

    Builds on ``extract_shadow_prices_full_ts`` (per-base-time-step
    normalized duals of the nodal energy balance), groups them by
    optimized year, and broadcasts each (carrier, node) price onto every
    conversion technology that has that carrier as an output carrier.

    Returns:
        pandas.DataFrame indexed by
        (``set_technologies``, ``set_output_carriers``, ``set_nodes``) with
        one column per year in ``set_time_steps_yearly`` holding the yearly
        average normalized shadow price. ``None`` if duals are unavailable.
    """
    full_ts = extract_shadow_prices_full_ts(optimization_setup)
    if full_ts is None:
        return None

    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    output_carriers_by_tech = sets["set_output_carriers"]

    time_steps = optimization_setup.energy_system.time_steps
    seq = np.asarray(time_steps.sequence_time_steps_operation)
    op2year = pd.Series(time_steps.time_steps_operation2year)
    base_year = pd.Series(seq).map(op2year)
    base_year.index = full_ts.columns

    # average per year per (carrier, node) -> wide frame indexed by (carrier, node)
    yearly = full_ts.T.groupby(base_year).mean().T
    yearly.columns.name = "year"

    # broadcast each (carrier, node) price onto every (tech, output_carrier, node)
    tech_carrier = pd.DataFrame(
        [(t, c) for t in techs for c in output_carriers_by_tech[t]],
        columns=["set_technologies", "set_output_carriers"],
    )
    yearly_long = yearly.reset_index().melt(
        id_vars=yearly.index.names,
        var_name="year",
        value_name="shadow_price",
    )
    merged = tech_carrier.merge(
        yearly_long,
        left_on="set_output_carriers",
        right_on="set_carriers",
        how="inner",
    )
    return merged.pivot_table(
        index=["set_technologies", "set_output_carriers", "set_nodes"],
        columns="year",
        values="shadow_price",
    )


def calculate_revenue(
    optimization_setup,
    capacity_addition_gw: float = 1.0,
    investment_year: int = 0,
) -> pd.Series:
    """Discounted lifetime revenue for a hypothetical capacity addition.

    Per (conversion technology, output carrier, node), the formula is::

        revenue = capacity_addition_gw
                  * specific_production(prev_year)
                  * Σ_{offset=0..L-1} (
                        shadow_price(year_y) * market_value
                        / (1 + r) ** (Δy * offset)
                    )

    where ``Δy = system.interval_between_years``, ``L`` is the technology
    lifetime in calendar years (rounded up to whole optimized years), and
    ``year_y = min(investment_year + offset, last_horizon_year)`` — i.e. the
    last available year's price is used once the lifetime extends past the
    optimization horizon.

    Specific production is taken from the most recent year solved in the
    optimization (the "previous year") and held constant over the lifetime.

    Args:
        optimization_setup: A solved ``OptimizationSetup``.
        capacity_addition_gw: Hypothetical capacity addition (default 1 GW).
        investment_year: Year index in ``set_time_steps_yearly`` in which the
            capacity is added.

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_output_carriers``, ``set_nodes``) with
        the discounted revenue in the model's money unit.
    """
    sets = optimization_setup.sets
    system = optimization_setup.system
    interval = system.interval_between_years
    horizon_years = sorted(sets["set_time_steps_yearly"])
    last_year = horizon_years[-1]

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    market_value = get_market_value(optimization_setup)

    # "previous year" = the most recent year solved in the optimization
    prev_year = last_year
    spec_prod = get_yearly_production(optimization_setup, year=prev_year)
    prices = get_shadow_price(optimization_setup)
    if spec_prod is None or prices is None or spec_prod.empty:
        return pd.Series(dtype=float, name="discounted_revenue")

    logging.info(
        "\n--- Revenue calculation inputs ---\n"
        f"Discount rates per (tech, node):\n{discount_rate}\n\n"
        f"Lifetimes per tech (years):\n{lifetime}\n\n"
        f"Market value per (tech, node):\n{market_value}\n\n"
        f"Specific yearly production "
        f"per (tech, output_carrier, node), year={prev_year}:\n{spec_prod}\n\n"
        f"Shadow prices per (tech, output_carrier, node) and year:\n{prices}\n"
    )

    revenue = pd.Series(0.0, index=spec_prod.index, name="discounted_revenue")
    for (tech, carrier, node), prod in spec_prod.items():
        if pd.isna(prod) or pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        r = float(discount_rate.loc[(tech, node)])
        mv = float(market_value.loc[(tech, node)])
        total = 0.0
        for year in range(tech_lifetime):
            try:
                price = float(prices.loc[(tech, carrier, node)])
            except KeyError:
                price = 0.0
            disc = (1 + r) ** (year)
            total += capacity_addition_gw * prod * price * mv / disc
        revenue.loc[(tech, carrier, node)] = total
    return revenue


