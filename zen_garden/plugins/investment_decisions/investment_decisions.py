"""Helpers for extracting investor-relevant signals from a solved optimization."""

import logging

import numpy as np
import pandas as pd


#helper functions
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

# flow, production and shadow price calculations
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

def get_flow_reference_carrier(optimization_setup) -> pd.Series:
    """Extracts the flow of the reference carrier for all conversion technologies and nodes from the optimization setup and divides it by the installed capacity.

    The reference carrier may be an input or output carrier
    (stored in ``sets["set_reference_carriers"]``).  Both
    ``flow_conversion_output`` and ``flow_conversion_input`` are checked.

    For each conversion technology, node and aggregated operational time step
    ``t``, returns::

        flow_ref[t] * time_steps_operation_duration[t] / capacity[year(t)]

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_nodes``, ``set_time_steps_operation``).
        ``NaN`` where capacity is zero.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_level = "set_time_steps_operation"

    ref_carriers = sets["set_reference_carriers"]  # dict: tech -> [carrier]

    def _load_flow(var_name, carrier_level):
        s = optimization_setup.model.solution[var_name].to_series().dropna()
        s = s[s.index.get_level_values("set_conversion_technologies").isin(techs)]
        s.index = s.index.rename({"set_conversion_technologies": "set_technologies"})
        if "set_location" in s.index.names:
            s.index = s.index.rename({"set_location": "set_nodes"})
        return s, carrier_level

    flow_out, out_level = _load_flow("flow_conversion_output", "set_output_carriers")
    flow_in, in_level = _load_flow("flow_conversion_input", "set_input_carriers")

    segments = []
    tech_ref_map: dict[str, str] = {}
    for tech in techs:
        ref = ref_carriers[tech][0]

        # Try output carriers first, then input carriers
        for flow_series, carrier_level in [(flow_out, out_level), (flow_in, in_level)]:
            tech_mask = flow_series.index.get_level_values("set_technologies") == tech
            carrier_mask = flow_series.index.get_level_values(carrier_level) == ref
            segment = flow_series[tech_mask & carrier_mask]
            if not segment.empty:
                segments.append(segment.droplevel(carrier_level))
                tech_ref_map[tech] = ref
                break

    if not segments:
        return pd.Series(dtype=float, name="specific_ref_flow_per_gw")

    ref_flow = pd.concat(segments)

    # Weight by duration: [energy/time] → [energy] per time-step interval
    ref_flow = ref_flow.mul(ref_flow.index.get_level_values(op_level).map(durations))

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[capacity.index.get_level_values("set_technologies").isin(techs)]
    if "set_location" in capacity.index.names:
        capacity.index = capacity.index.rename({"set_location": "set_nodes"})

    flow_df = ref_flow.to_frame("flow_energy").reset_index()
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
    merged["reference_carrier"] = merged["set_technologies"].map(tech_ref_map)
    result = merged.set_index(
        ["set_technologies", "reference_carrier", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_ref_flow_per_gw"

    ref_summary = "\n".join(f"  {t}: {c}" for t, c in tech_ref_map.items())
    print(
        f"\n--- Specific reference carrier flow per GW [MWh/GW] ---\n"
        f"Reference carriers used:\n{ref_summary}\n\n"
        f"{result}\n"
    )
    return result

def get_shadow_price(optimization_setup) -> pd.Series | None:
    """Per-aggregated-time-step shadow prices for output carriers of all conversion techs.

    Transforms to pd.Series and maps  duals variables / shadow prices to technologies (based on output carriers) istead of technologies.
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


# revenue and cost calculations
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

def get_capex(optimization_setup) -> pd.Series:
    """Total annualized CAPEX [MEUR] for the capacity additions returned by
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

def get_fixed_opex_discounted(optimization_setup) -> pd.Series:
    """Total fixed OPEX [MEUR] for the capacity additions  returned by
    ``_capacity_addition`` discounted to the decision year, for conversion technologies only.

    Extrects fixed OPEX data from ``optimization_setup`` and uses the same capacity addition and investment delay principle as in the revenue calculation. 

    Dataflow:
    - extract fixed OPEX specific to conversion technologies for all years available in the dataset. For years where no data is available, the value for the latest available year is used.
    - apply the same capacity_addition and investment delay principle as in the revenue calculation to calculate the discounted fixed OPEX for each technology and node.
    - The fixed OPEX are discounted to the decision year using the discount rate from ``get_discount_rate`` and the investment delay from ``_get_investment_delay`` over the lifetime of the respective technology.
    - The result is printed and returned as a pandas Series indexed by (``set_conversion_technologies``, ``set_nodes``) with the discounted fixed OPEX in model money units.
      """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    base_year = max(sets["set_time_steps_yearly"])

    # Sum over capacity types; rename set_location → set_nodes
    opex_series = (
        parameters.opex_specific_fixed
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    if "set_location" in opex_series.index.names:
        opex_series.index = opex_series.index.rename({"set_location": "set_nodes"})
    opex_series = opex_series[
        opex_series.index.get_level_values("set_technologies").isin(techs)
    ]

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    idx = pd.MultiIndex.from_product(
        [techs, nodes],
        names=["set_conversion_technologies", "set_nodes"],
    )
    result = pd.Series(0.0, index=idx, name="fixed_opex_discounted")

    for tech in techs:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        cap = float(capacity_addition[tech])

        for node in nodes:
            r = float(discount_rate.loc[(tech, node)])
            total = 0.0

            opex_val = float(opex_series.loc[(tech, node, base_year)])            

            for offset in range(delay, tech_lifetime + delay):
                total += opex_val / (1 + r) ** offset

            result.loc[(tech, node)] = cap * total

    print(f"\n--- Discounted Fixed OPEX for Capacity Additions ---\n{result}\n")
    return result

def get_variable_opex_discounted(optimization_setup) -> pd.Series:
    """Total variable OPEX [MEUR] for the capacity additions  returned by
    ``_capacity_addition`` discounted to the decision year, for conversion technologies only.

    Extracts variable OPEX data from ``optimization_setup`` and uses the same capacity addition and investment delay principle as in the revenue calculation. 

    Dataflow:
    - extract variable OPEX [EUR/MWh] specific to conversion technologies for all years available in the dataset. For years where no data is available, the value for the latest available year is used.
    - Obtain the actual cost by multiplying the variable OPEX with the specific production per GW installed capacity and the capacity addition for each technology and node. 
    - apply the same investment delay principle as in the revenue calculation to calculate the discounted variable OPEX for each technology and node.
    - The variable OPEX are discounted to the decision year using the discount rate from ``get_discount_rate`` and the investment delay from ``_get_investment_delay`` over the lifetime of the respective technology.
    - The result is printed and returned as a pandas Series indexed by (``set_conversion_technologies``, ``set_nodes``) with the discounted variable OPEX in model money units.
      """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    op_level = "set_time_steps_operation"
    base_year = max(sets["set_time_steps_yearly"])

    # Extract opex_specific_variable [EUR/MWh] per operational time step
    opex_var = parameters.opex_specific_variable.to_series().dropna()
    opex_var = opex_var[opex_var.index.get_level_values("set_technologies").isin(techs)]
    if "set_location" in opex_var.index.names:
        opex_var.index = opex_var.index.rename({"set_location": "set_nodes"})

    # Specific reference flow per GW [MWh/GW per time step]
    ref_flow = get_flow_reference_carrier(optimization_setup)
    ref_flow = ref_flow.droplevel("reference_carrier")

    # Multiply: [EUR/MWh] * [MWh/GW] = [EUR/GW] per time step, then sum per year
    opex_var_df = opex_var.rename("opex_var").reset_index()
    ref_flow_df = ref_flow.rename("ref_flow").reset_index()
    product_df = opex_var_df.merge(
        ref_flow_df,
        on=["set_technologies", "set_nodes", op_level],
        how="inner",
    )
    product_df["cost"] = product_df["opex_var"] * product_df["ref_flow"]
    product_df["set_time_steps_yearly"] = product_df[op_level].map(op2year)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        print(f"\n--- product_df (alle Einträge) ---\n{product_df}\n")
    annual_var_opex = (
        product_df
        .groupby(["set_technologies", "set_nodes", "set_time_steps_yearly"])["cost"]
        .sum()
    )

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    idx = pd.MultiIndex.from_product(
        [techs, nodes],
        names=["set_conversion_technologies", "set_nodes"],
    )
    result = pd.Series(0.0, index=idx, name="variable_opex_discounted")

    for tech in techs:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        cap = float(capacity_addition[tech])

        for node in nodes:
            r = float(discount_rate.loc[(tech, node)])
            total = 0.0

            opex_val = float(annual_var_opex.loc[(tech, node, base_year)])
            

            for offset in range(delay, tech_lifetime + delay):
                total += opex_val / (1 + r) ** offset

            result.loc[(tech, node)] = cap * total

    print(f"\n--- Discounted Variable OPEX for Capacity Additions ---\n{result}\n")
    return result


# profitability calculation
def calculate_profitability(optimization_setup) -> pd.Series:
    """Calculate profitability of the capacity addition as revenue minus costs.

    Costs include both CAPEX and OPEX (fixed and variable).  The same capacity
    addition, investment delay and discounting principles are applied as in the
    revenue and cost calculations.

    Returns:
        pandas.Series indexed by (``set_conversion_technologies``, ``set_nodes``)
        with the discounted profitability in model money units.
    """
    revenue = calculate_revenue(optimization_setup)
    capex = get_capex(optimization_setup)
    fixed_opex = get_fixed_opex_discounted(optimization_setup)
    variable_opex = get_variable_opex_discounted(optimization_setup)

    # revenue is indexed by (set_technologies, set_output_carriers, set_nodes);
    # sum over output carriers and rename to match the cost index names.
    revenue_by_tech_node = (
        revenue
        .groupby(level=["set_technologies", "set_nodes"])
        .sum()
        .rename_axis(index={"set_technologies": "set_conversion_technologies"})
    )

    profitability = revenue_by_tech_node.subtract(capex, fill_value=0)
    profitability = profitability.subtract(fixed_opex, fill_value=0)
    profitability = profitability.subtract(variable_opex, fill_value=0)
    profitability.name = "profitability"

    print(
        f"\n--- Profitability of Capacity Additions ---\n"
        f"{'Tech / Node':<45} {'Revenue':>12} {'CAPEX':>12} "
        f"{'Fixed OPEX':>12} {'Var OPEX':>12} {'Profit':>12}\n"
        + "-" * 105
    )
    for idx in profitability.index:
        rev_val = revenue_by_tech_node.get(idx, 0.0)
        cap_val = capex.get(idx, 0.0)
        fop_val = fixed_opex.get(idx, 0.0)
        vop_val = variable_opex.get(idx, 0.0)
        pro_val = profitability[idx]
        print(
            f"{str(idx):<45} {rev_val:>12.2f} {cap_val:>12.2f} "
            f"{fop_val:>12.2f} {vop_val:>12.2f} {pro_val:>12.2f}"
        )
    print()
    return profitability


#visualization
def visualization(
    optimization_setup,
    output_dir: str = r"D:\Students\ssambale_jwiegner\Crystal-Ball\visualization",
) -> None:
    """Generate and save investment-decision visualizations.

    Saves two plot types to ``output_dir``, each suffixed with the current
    calendar year so runs for different optimization periods do not overwrite
    each other:
    - ``profitability_breakdown_<year>.png``: revenue vs. stacked costs + net-profit marker per tech/node
    - ``profitability_net_<year>.png``: net profit bar chart, color-coded profitable / loss
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    year = max(optimization_setup.sets["set_time_steps_yearly"])

    # --- collect component data ---
    revenue = calculate_revenue(optimization_setup)
    capex = get_capex(optimization_setup)
    fixed_opex = get_fixed_opex_discounted(optimization_setup)
    variable_opex = get_variable_opex_discounted(optimization_setup)

    revenue_by_tech_node = (
        revenue
        .groupby(level=["set_technologies", "set_nodes"])
        .sum()
        .rename_axis(index={"set_technologies": "set_conversion_technologies"})
    )

    profitability = (
        revenue_by_tech_node
        .subtract(capex, fill_value=0)
        .subtract(fixed_opex, fill_value=0)
        .subtract(variable_opex, fill_value=0)
    )
    profitability.name = "profitability"

    idx = profitability.index
    if idx.empty:
        print("visualization: keine Profitabilitätsdaten vorhanden, Diagramme werden übersprungen.")
        return

    rev = revenue_by_tech_node.reindex(idx, fill_value=0)
    cap = capex.reindex(idx, fill_value=0)
    fop = fixed_opex.reindex(idx, fill_value=0)
    vop = variable_opex.reindex(idx, fill_value=0)
    pro = profitability

    labels = [f"{t}\n{n}" for t, n in idx]
    x = list(range(len(labels)))
    fig_width = max(9, len(labels) * 1.6)

    # ------------------------------------------------------------------ #
    # Plot 1 – Revenue vs. Costs Breakdown + Net Profit Marker            #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    w = 0.45

    ax.bar(x, rev.values, w, label="Revenue", color="#2ecc71", zorder=3)
    ax.bar(x, -cap.values, w, label="CAPEX", color="#e74c3c", zorder=3)
    ax.bar(x, -fop.values, w, label="Fixed OPEX", color="#e67e22",
           bottom=-cap.values, zorder=3)
    ax.bar(x, -vop.values, w, label="Var. OPEX", color="#f39c12",
           bottom=(-cap - fop).values, zorder=3)
    ax.plot(x, pro.values, "D", color="#2c3e50", markersize=9,
            label="Net Profit", zorder=4, clip_on=False)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Discounted value [model money units]")
    ax.set_title(f"Investment Profitability Breakdown per Technology / Node ({year})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    p1 = out / f"profitability_breakdown_{year}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1}")

    # ------------------------------------------------------------------ #
    # Plot 2 – Net Profitability Bar Chart                                #
    # ------------------------------------------------------------------ #
    bar_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in pro.values]
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(x, pro.values, color=bar_colors, edgecolor="white", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

    value_range = max(abs(pro.values)) if len(pro) else 1
    offset = value_range * 0.02
    for bar, val in zip(bars, pro.values):
        y = val + offset if val >= 0 else val - offset
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:,.0f}", ha="center", va=va, fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Net Discounted Profit [model money units]")
    ax.set_title(f"Net Investment Profitability per Technology / Node ({year})")
    handles = [
        mpatches.Patch(color="#27ae60", label="Profitable"),
        mpatches.Patch(color="#c0392b", label="Loss"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    p2 = out / f"profitability_net_{year}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2}")

