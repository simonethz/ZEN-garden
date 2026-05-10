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


def extract_aggregated_dual(optimization_setup):
    """Return the dual variables / shadow prices for aggregated time steps."""
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

     # divide by per-year annuity, matching `Results.get_full_ts` for duals.
    time_steps = optimization_setup.energy_system.time_steps
    annuity = _normalize_interval(optimization_setup)
    op2year = pd.Series(time_steps.time_steps_operation2year)
    annuity_per_op = op2year.reindex(df_duals.columns).map(annuity)
    df_duals = df_duals.div(annuity_per_op, axis=1)

    return df_duals

def extract_normalized_dual(optimization_setup):
    """Return the dual variables / shadow prices for aggregated time steps, normalized to per-hour values."""
    df_duals = extract_aggregated_dual(optimization_setup)
    time_steps = optimization_setup.energy_system.time_steps

    # divide by duration: the raw dual equals duration * per-hour price because
    # operational variables enter the objective weighted by
    # `time_steps_operation_duration`.
    durations = pd.Series(time_steps.time_steps_operation_duration).reindex(
        df_duals.columns
    )
    df_duals = df_duals.div(durations, axis=1)
   
    logging.info(
        "\n--- Normalized shadow prices per aggregated operational time step ---\n"
        f"{df_duals}\n"
    )
    return df_duals

def extract_shadow_prices_full_ts(optimization_setup):
    """Return the dual variables / shadow prices expanded to the full base time step resolution."""
    time_steps = optimization_setup.energy_system.time_steps
    df_duals = extract_normalized_dual(optimization_setup)
    # map aggregated operation time steps onto the underlying base time steps
    sequence = np.asarray(time_steps.sequence_time_steps_operation)
    full_ts = df_duals.reindex(columns=sequence)
    full_ts.columns = pd.RangeIndex(len(sequence), name="base_time_step")

    return full_ts

def extract_average_shadow_prices(optimization_setup):
    """Return the average shadow price of nodal energy balance duals across time.
    """
    full_ts = extract_shadow_prices_full_ts(optimization_setup)
    if full_ts is None:
        return None

    average_duals = full_ts.mean(axis=1).to_frame(name="average_shadow_price")
    return average_duals


# revenue calculation


def get_lifetime(optimization_setup) -> pd.Series:
    """Lifetime per conversion technology, in calendar years.

    Returns:
        pandas.Series indexed by ``set_technologies`` (restricted to conversion
        technologies) holding the lifetime parameter from the model.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])    
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


def get_specific_production(optimization_setup) -> pd.Series:
    """Production per GW installed capacity per aggregated operational time step.

    For each conversion technology, output carrier, node and aggregated
    operational time step ``t``, returns::

        flow_conversion_output[t] * time_steps_operation_duration[t]
            / capacity[year(t)]

    Units: energy per capacity (GWh/GW = h), i.e. full-load-equivalent
    hours per GW for that time step. Summing the values for all time steps
    of one optimized year therefore yields the yearly production per GW
    (in hours of full-load equivalent / GWh per GW).

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_output_carriers``, ``set_nodes``,
        ``set_time_steps_operation``). ``NaN`` where capacity is zero.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_level = "set_time_steps_operation"

    flow = (
        optimization_setup.model.solution["flow_conversion_output"]
        .to_series()
        .dropna()
    )
    flow = flow[
        flow.index.get_level_values("set_conversion_technologies").isin(techs)
    ]
    flow.index = flow.index.rename(
        {"set_conversion_technologies": "set_technologies"}
    )
    # weight each flow value by the duration of its aggregated time step
    # (dimensional from [energy/time] to [energy] per time-step interval)
    flow = flow.mul(flow.index.get_level_values(op_level).map(durations))

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[
        capacity.index.get_level_values("set_technologies").isin(techs)
    ]
    capacity.index = capacity.index.rename({"set_location": "set_nodes"})

    flow_df = flow.to_frame("flow_energy").reset_index()
    flow_df["set_time_steps_yearly"] = flow_df[op_level].map(op2year)
    cap_df = capacity.to_frame("capacity").reset_index()

    merged = flow_df.merge(
        cap_df,
        on=["set_technologies", "set_nodes", "set_time_steps_yearly"],
        how="left",
    )
    merged["spec"] = (
        (merged["flow_energy"] / merged["capacity"])
        .replace([np.inf, -np.inf], np.nan)
    )
    result = merged.set_index(
        ["set_technologies", "set_output_carriers", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_production_per_gw"
    return result


def get_shadow_price(optimization_setup) -> pd.Series | None:
    """Per-aggregated-time-step shadow prices for output carriers of all conversion techs.

    Broadcasts each ``(carrier, node, time_step_operation)`` normalized
    shadow price onto every conversion technology whose set of output
    carriers contains that carrier.

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_output_carriers``, ``set_nodes``,
        ``set_time_steps_operation``) with the per-hour shadow price, or
        ``None`` if duals are unavailable.
    """
    df = extract_normalized_dual(optimization_setup)
    if df is None:
        return None
    series = df.stack().dropna()
    series.name = "shadow_price"

    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    output_carriers_by_tech = sets["set_output_carriers"]

    long = series.reset_index()
    tech_carrier = pd.DataFrame(
        [(t, c) for t in techs for c in output_carriers_by_tech[t]],
        columns=["set_technologies", "set_output_carriers"],
    )
    merged = tech_carrier.merge(
        long,
        left_on="set_output_carriers",
        right_on="set_carriers",
        how="inner",
    )
    return merged.set_index(
        [
            "set_technologies",
            "set_output_carriers",
            "set_nodes",
            "set_time_steps_operation",
        ]
    )["shadow_price"]


def calculate_revenue(
    optimization_setup,
    capacity_addition_gw: float = 1.0,
    investment_year: int = 0,
) -> pd.Series:
    """Discounted lifetime revenue for a hypothetical capacity addition.

    Per (conversion technology, output carrier, node), the formula is::

        revenue = capacity_addition_gw
                  * Σ_{offset=0..n_steps-1} (
                        annual_revenue[tech, oc, node, eff_year]
                        / (1 + r) ** offset
                    )

        where annual_revenue[tech, oc, node, year]
            = Σ_{t in time_steps(year)} (
                  shadow_price[oc, node, t] * specific_production[tech, oc, node, t]
              )

    ``n_steps = ceil(lifetime / Δy)`` where ``Δy =
    system.interval_between_years``. ``eff_year = min(investment_year +
    offset, last_horizon_year)`` forward-fills the annual revenue of the last
    horizon year once the lifetime extends past the optimization horizon. The
    aggregated-time-step duration is already baked into ``specific_production``
    (units GWh/GW), so no explicit ``duration[t]`` factor appears here.

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
    spec_prod = get_specific_production(optimization_setup)
    prices = get_shadow_price(optimization_setup)
    if spec_prod is None or prices is None or spec_prod.empty:
        return pd.Series(dtype=float, name="discounted_revenue")

    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)

    # Pre-aggregate spec_prod * shadow_price over all t within each year.
    op_level = "set_time_steps_operation"
    product = spec_prod.mul(prices).dropna()
    product.name = "prod_revenue"
    product_df = product.reset_index()
    product_df["set_time_steps_yearly"] = product_df[op_level].map(op2year)
    annual_rev = (
        product_df.groupby(
            ["set_technologies", "set_output_carriers", "set_nodes", "set_time_steps_yearly"]
        )["prod_revenue"].sum()
    )

    logging.info(
        "\n--- Revenue calculation inputs ---\n"
        f"Discount rates per (tech, node):\n{discount_rate}\n\n"
        f"Lifetimes per tech (years):\n{lifetime}\n\n"
        f"Annual revenue (spec_prod * shadow_price summed over t) per "
        f"(tech, oc, node, year):\n{annual_rev}\n"
    )

    group_index = annual_rev.index.droplevel("set_time_steps_yearly").unique()

    revenue = pd.Series(0.0, index=group_index, name="discounted_revenue")
    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        n_steps = int(np.ceil(tech_lifetime / interval))
        r = float(discount_rate.loc[(tech, node)])
        total = 0.0
        for offset in range(n_steps):
            eff_year = min(investment_year + offset, last_year)
            disc = (1 + r) ** offset
            key = (tech, carrier, node, eff_year)
            if key not in annual_rev.index:
                continue
            val = annual_rev.loc[key]
            if pd.isna(val):
                continue
            total += float(val) / disc
        revenue.loc[(tech, carrier, node)] = capacity_addition_gw * total
    return revenue


def get_specific_capex(optimization_setup) -> dict[str, pd.Series]:
    """Specific CAPEX per GW (or GWh) for new capacity additions.

    Pulls the linear specific-capex coefficients used by the optimizer when
    adding new capacity, restricted to the most recent year in
    ``set_time_steps_yearly``:

    - ``capex_specific_conversion`` — money per GW for conversion technologies
      modeled with a linear capex (i.e. those in ``set_capex_linear``;
      PWA-capex technologies are not represented here).
    - ``capex_specific_storage`` — money per GW for storage power capacity and
      money per GWh for storage energy capacity. The unit is implied by the
      ``set_capacity_types`` index level (typically "power" / "energy").

    Transport technologies are intentionally excluded since they are indexed
    by edges rather than nodes.

    Args:
        optimization_setup: An ``OptimizationSetup`` whose parameters have
            been initialized.

    Returns:
        dict with keys ``"conversion"`` and ``"storage"``. Each value is a
        ``pandas.Series`` indexed by the parameter's location-and-tech levels
        for the latest year. Keys are omitted if the corresponding parameter
        is not present in the model.
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters
    latest_year = max(sets["set_time_steps_yearly"])

    def _select_year(param) -> pd.Series:
        # parameter dim names get re-aliased (e.g. set_time_steps_yearly -> year);
        # convert to a Series first and then filter on whichever year level exists.
        series = param.to_series().dropna()
        for level_name in ("set_time_steps_yearly", "year"):
            if level_name in series.index.names:
                return series.xs(latest_year, level=level_name)
        return series

    output: dict[str, pd.Series] = {}

    if hasattr(parameters, "capex_specific_conversion"):
        conv = _select_year(parameters.capex_specific_conversion)
        conv.name = "capex_specific_conversion"
        output["conversion"] = conv

    if hasattr(parameters, "capex_specific_storage"):
        stor = _select_year(parameters.capex_specific_storage)
        stor.name = "capex_specific_storage"
        output["storage"] = stor

    logging.info(
        f"\n--- Specific CAPEX for new capacity additions, year {latest_year} ---"
    )
    if "conversion" in output:
        logging.info(
            "\nConversion technologies (money per GW, linear-capex techs only):\n"
            f"{output['conversion']}\n"
        )
    if "storage" in output:
        logging.info(
            "\nStorage technologies "
            "(money per GW for 'power', per GWh for 'energy'):\n"
            f"{output['storage']}\n"
        )

    return output
