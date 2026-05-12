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

# dual extraction
def extract_aggregated_dual(optimization_setup):
    """Return the dual variables / shadow prices for aggregated time steps (e.g. 5 time steps, values = hourly prices * number of this time step/year)."""
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
    """Return the dual variables / shadow prices for aggregated time steps, normalized to per-hour values (e.g. 5 time steps, values = hourly prices)."""
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
    """Return the dual variables / shadow prices expanded to the full base time step resolution (e.g. 8760 values, values = hourly prices)."""
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
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])    
    lifetime = optimization_setup.parameters.lifetime.to_series()
    lifetime.name = "lifetime"
    lt = lifetime.reindex(techs)
    return lt


def get_discount_rate(optimization_setup) -> pd.Series:
    """Discount rate per (conversion technology, node).

    The model currently has only a system-wide scalar ``discount_rate``, so
    every (tech, node) pair receives the same value. 
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    rate = float(np.asarray(optimization_setup.parameters.discount_rate).item())
    idx = pd.MultiIndex.from_product(
        [techs, nodes], names=["set_technologies", "set_nodes"]
    )
    discount_rate = pd.Series(rate, index=idx, name="discount_rate")
    return discount_rate


def get_specific_production(optimization_setup) -> pd.Series:
    """Production per GW installed capacity per aggregated operational time step. Sum over all time steps of one optimized year yields the yearly production per GW (in hours of full-load equivalent / GWh per GW).

    For each conversion technology, output carrier, node and aggregated
    operational time step ``t``, returns::

        flow_conversion_output[t] * time_steps_operation_duration[t]
            / capacity[year(t)]
    
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
    mapped_shadow_price = merged.set_index(
        [
            "set_technologies",
            "set_output_carriers",
            "set_nodes",
            "set_time_steps_operation",
        ]
    )["shadow_price"]
    return mapped_shadow_price


def calculate_revenue(optimization_setup) -> pd.Series:
    """Discounted lifetime revenue for a hypothetical capacity addition.

    Per (conversion technology, output carrier, node), the formula is::

        revenue = capacity_addition_gw
                  * Σ_{n=delay..lifetime+delay-1} (
                        annual_revenue[tech, oc, node]
                        / (1 + r) ** n
                    )

        where annual_revenue[tech, oc, node]
            = Σ_{t in aggregated time steps} (
                  shadow_price[oc, node, t] * specific_production[tech, oc, node, t]
              )

    The single optimization year is used as a representative annual revenue
    for all years of the technology lifetime. An optional investment delay
    (from ``_get_investment_delay``) shifts the discounting window forward.
    Capacity additions are taken from ``_capacity_addition``.

    Args:
        optimization_setup: A solved ``OptimizationSetup``.

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_output_carriers``, ``set_nodes``) with
        the discounted revenue in the model's money unit.
    """
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    spec_prod = get_specific_production(optimization_setup)
    prices = get_shadow_price(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition_gw = _capacity_addition(optimization_setup)
    if spec_prod is None or prices is None or spec_prod.empty:
        print("\n--- Revenue calculation skipped: missing specific production or shadow price data ---\n")
        return pd.Series(dtype=float, name="discounted_revenue")

    # Aggregate spec_prod * shadow_price over all operational time steps.
    product = spec_prod.mul(prices).dropna()
    product.name = "prod_revenue"
    product_df = product.reset_index()
    annual_rev = (
        product_df.groupby(
            ["set_technologies", "set_output_carriers", "set_nodes"]
        )["prod_revenue"].sum()
    )

    logging.info(
        "\n--- Revenue calculation inputs ---\n"
        f"Lifetimes per tech (years):\n{lifetime}\n\n"
        f"Specific production per (tech, oc, node, t):\n{spec_prod}\n\n"
        f"Annual revenue (spec_prod * shadow_price summed over t) per "
        f"(tech, oc, node):\n{annual_rev}\n"
    )

    group_index = annual_rev.index

    revenue = pd.Series(0.0, index=group_index, name="discounted_revenue")
    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = investment_delay[tech]
        capacity = capacity_addition_gw[tech]
        r = float(discount_rate.loc[(tech, node)])
        total = 0.0
        for offset in range(0 + delay,tech_lifetime + delay):
            disc = (1 + r) ** offset
            key = (tech, carrier, node)
            if key not in annual_rev.index:
                continue
            val = annual_rev.loc[key]
            if pd.isna(val):
                continue
            total += float(val) / disc
        revenue.loc[(tech, carrier, node)] = capacity * total
    return revenue

def _capacity_addition(optimization_setup, capacity_gw: float = 1.0) -> pd.Series:
    """Capacity addition per conversion technology.

    Currently returns a fixed ``capacity_gw`` for every technology.  The
    function is intentionally kept as a thin shell so that per-technology or
    per-node overrides can be introduced here without touching callers.
    """
    techs = list(optimization_setup.sets["set_conversion_technologies"])
    return pd.Series(
        capacity_gw,
        index=pd.Index(techs, name="set_conversion_technologies"),
        name="capacity_addition_gw",
    )


def _get_investment_delay(optimization_setup, delay_years: int = 0) -> pd.Series:
    """Investment delay (in years) per conversion technology.

    Currently returns a fixed ``delay_years`` for every technology.  The
    function is intentionally kept as a thin shell so that per-technology or
    per-node overrides can be introduced here without touching callers.
    """
    techs = list(optimization_setup.sets["set_conversion_technologies"])
    return pd.Series(
        delay_years,
        index=pd.Index(techs, name="set_conversion_technologies"),
        name="investment_delay_years",
    )


def get_capex(optimization_setup) -> pd.Series:
    """Total annualized CAPEX for the capacity additions returned by
    ``_capacity_addition``, for conversion technologies only.

    Reads CAPEX data directly from ``optimization_setup`` and supports both:

    - **Linear** (``set_capex_linear``):
      ``capex = capex_specific_conversion [money/GW] * capacity_addition [GW]``
    - **PWA** (``set_capex_pwa``): total cost is interpolated from the
      technology's piecewise-linear breakpoint curve.  The curve is
      node-independent, so every node of that technology receives the same
      value.

    Returns:
        ``pd.Series`` indexed by ``(set_conversion_technologies, set_nodes)``
        with the annualised CAPEX in model money units.  
    """
    from zen_garden.model.technology.conversion_technology import ConversionTechnology

    sets = optimization_setup.sets
    parameters = optimization_setup.parameters
    latest_year = max(sets["set_time_steps_yearly"])
    nodes = list(sets["set_nodes"])
    capacity = _capacity_addition(optimization_setup)

    def _select_year(param) -> pd.Series:
        series = param.to_series().dropna()
        for level_name in ("set_time_steps_yearly", "year"):
            if level_name in series.index.names:
                return series.xs(latest_year, level=level_name)
        return series

    records: list[dict] = []

    # Linear conversion technologies
    if hasattr(parameters, "capex_specific_conversion"):
        capex_specific = _select_year(parameters.capex_specific_conversion)
        for (tech, node), specific in capex_specific.items():
            if tech not in capacity.index:
                continue
            records.append(
                {
                    "set_conversion_technologies": tech,
                    "set_nodes": node,
                    "capex": float(specific) * float(capacity[tech]),
                }
            )

    # PWA conversion technologies (curve is node-independent → same value for all nodes)
    for tech in sets["set_conversion_technologies"]:
        if not optimization_setup.get_attribute_of_specific_element(
            ConversionTechnology, tech, "capex_is_pwa"
        ):
            continue
        pwa_data = optimization_setup.get_attribute_of_specific_element(
            ConversionTechnology, tech, "pwa_capex"
        )
        print(f"\n used PWA for " + tech)
        cap = float(capacity[tech])
        total_capex = float(
            np.interp(cap, pwa_data["capacity_addition"], pwa_data["capex"])
        )
        for node in nodes:
            records.append(
                {
                    "set_conversion_technologies": tech,
                    "set_nodes": node,
                    "capex": total_capex,
                }
            )

    if not records:
        result = pd.Series(dtype=float, name="capex")
    else:
        result = pd.DataFrame(records).set_index(
            ["set_conversion_technologies", "set_nodes"]
        )["capex"]
        result.name = "capex"

    print(f"\n--- CAPEX für Kapazitätszubau (annualisiert) ---\n{result}\n")
   
    return result

