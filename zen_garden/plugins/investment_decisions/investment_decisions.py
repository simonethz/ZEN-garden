"""Helpers for extracting investor-relevant signals from a solved optimization."""

import logging

import numpy as np
import pandas as pd


# timestamp of the current program run, shared across all optimization steps so
# that every ``visualization`` call of one run writes into the same folder.
_RUN_TIMESTAMP: str | None = None

def _get_run_timestamp() -> str:
    """Return a single timestamp string for the whole program run.

    Computed once on first use and cached at module level, so repeated
    ``visualization`` calls (one per rolling-horizon / optimization step) all
    share the same folder name.
    """
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        from datetime import datetime

        _RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _RUN_TIMESTAMP

def _get_output_dir(optimization_setup) -> str:
    """Visualization directory of the dataset currently being optimized.

    Derived from the dataset input folder ``analysis.dataset``
    (``.../<dataset_root>/data/<name>``) so that figures and history stores land
    in ``<dataset_root>/visualization`` of the *respective* dataset instead of a
    hard-coded path. Mirrors the output location used in ``run_and_visualize``.
    """
    from pathlib import Path

    dataset = Path(optimization_setup.analysis.dataset)
    return str(dataset.parent.parent / "visualization")

def _get_run_output_dir(optimization_setup):
    """Timestamped per-run subfolder under the dataset's visualization dir.

    Created on demand and shared by every figure / CSV of one program run.
    """
    from pathlib import Path

    out = Path(_get_output_dir(optimization_setup)) / _get_run_timestamp()
    out.mkdir(parents=True, exist_ok=True)
    return out


# general helper functions
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
    print("\n--- Average shadow prices across time steps ---")
    print(average_duals)
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
    merged["spec"] = merged["flow_energy"] / merged["capacity"]
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
    merged["spec"] = merged["flow_energy"] / merged["capacity"]
    merged["reference_carrier"] = merged["set_technologies"].map(tech_ref_map)
    result = merged.set_index(
        ["set_technologies", "reference_carrier", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_ref_flow_per_gw"

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
 
def get_flow_input_carrier(optimization_setup) -> pd.Series:
    """Input carrier flow per GW installed capacity per aggregated operational time step.

    For each conversion technology, input carrier, node and aggregated
    operational time step ``t``, returns::

        flow_conversion_input[t] * time_steps_operation_duration[t]
            / capacity[year(t)]

    Technologies without an input carrier do not appear in the index; callers
    should treat missing entries as 0.

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_input_carriers``, ``set_nodes``,
        ``set_time_steps_operation``). 0 where capacity is zero.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    durations = pd.Series(time_steps.time_steps_operation_duration)
    op_level = "set_time_steps_operation"

    flow = optimization_setup.model.solution["flow_conversion_input"].to_series().dropna()
    flow = flow[flow.index.get_level_values("set_conversion_technologies").isin(techs)]
    flow.index = flow.index.rename({"set_conversion_technologies": "set_technologies"})
    if "set_location" in flow.index.names:
        flow.index = flow.index.rename({"set_location": "set_nodes"})

    if flow.empty:
        return pd.Series(dtype=float, name="specific_input_flow_per_gw")

    # Weight by duration: [energy/time] → [energy] per time-step interval
    flow = flow.mul(flow.index.get_level_values(op_level).map(durations))

    capacity = (
        optimization_setup.model.solution["capacity"]
        .sum("set_capacity_types")
        .to_series()
        .dropna()
    )
    capacity = capacity[capacity.index.get_level_values("set_technologies").isin(techs)]
    if "set_location" in capacity.index.names:
        capacity.index = capacity.index.rename({"set_location": "set_nodes"})

    flow_df = flow.to_frame("flow_energy").reset_index()
    flow_df["set_time_steps_yearly"] = flow_df[op_level].map(op2year)
    cap_df = capacity.to_frame("capacity").reset_index()

    merged = flow_df.merge(
        cap_df,
        on=["set_technologies", "set_nodes", "set_time_steps_yearly"],
        how="left",
    )
    merged["spec"] = merged["flow_energy"] / merged["capacity"]
    merged["spec"] = merged["spec"].fillna(0.0)
    result = merged.set_index(
        ["set_technologies", "set_input_carriers", "set_nodes", op_level]
    )["spec"]
    result.name = "specific_input_flow_per_gw"
    return result



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
    """Total upfront (overnight) investment cost [MEUR] for the capacity additions returned by
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
        with the upfront investment cost in model money units.
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

   
    return result

def get_fixed_opex_discounted(optimization_setup) -> pd.Series:
    """Total fixed OPEX [MEUR] for the capacity additions  returned by
    ``_capacity_addition`` discounted to the decision year, for conversion technologies only.

    Extracts fixed OPEX data from ``optimization_setup`` and uses the same capacity addition and investment delay principle as in the revenue calculation.

    Dataflow:
    - extract the fixed OPEX of the decision year (the latest year in ``set_time_steps_yearly``); this value is assumed constant for every year of the technology lifetime.
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

            try:
                opex_val = float(opex_series.loc[(tech, node, base_year)])
            except KeyError:
                continue

            for offset in range(delay, tech_lifetime + delay):
                total += opex_val / (1 + r) ** offset

            result.loc[(tech, node)] = cap * total

    return result

def get_variable_opex_discounted(optimization_setup) -> pd.Series:
    """Total variable OPEX [MEUR] for the capacity additions  returned by
    ``_capacity_addition`` discounted to the decision year, for conversion technologies only.

    Extracts variable OPEX data from ``optimization_setup`` and uses the same capacity addition and investment delay principle as in the revenue calculation. 

    Dataflow:
    - extract the variable OPEX [EUR/MWh] of the decision year (the latest year in ``set_time_steps_yearly``); this value is assumed constant for every year of the technology lifetime.
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

            try:
                opex_val = float(annual_var_opex.loc[(tech, node, base_year)])
            except KeyError:
                continue

            for offset in range(delay, tech_lifetime + delay):
                total += opex_val / (1 + r) ** offset

            result.loc[(tech, node)] = cap * total

    return result

def calculate_input_carrier_cost(optimization_setup) -> pd.Series:
    """Discounted lifetime input carrier cost for a hypothetical capacity addition.

    Analogous to ``calculate_revenue`` but for input carriers instead of output
    carriers. The shadow price of each input carrier at each node serves as the
    fuel/energy price.

    Per (conversion technology, input carrier, node), the formula is::

        cost = capacity_addition_gw
               * Σ_{n=delay..lifetime+delay-1} (
                     annual_cost[tech, ic, node]
                     / (1 + r) ** n
                 )

        where annual_cost[tech, ic, node]
            = Σ_{t in aggregated time steps} (
                  shadow_price[ic, node, t]
                  * specific_input_flow[tech, ic, node, t]
              )

    Returns:
        pandas.Series indexed by
        (``set_technologies``, ``set_input_carriers``, ``set_nodes``) with
        the discounted input carrier cost in the model's money unit.
    """
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition_gw = _capacity_addition(optimization_setup)

    input_flow = get_flow_input_carrier(optimization_setup)
    if input_flow.empty:
        print("\n--- Input carrier cost skipped: no input flow data ---\n")
        return pd.Series(dtype=float, name="discounted_input_carrier_cost")

    duals = extract_normalized_dual(optimization_setup)
    if duals is None:
        print("\n--- Input carrier cost skipped: no dual variables available ---\n")
        return pd.Series(dtype=float, name="discounted_input_carrier_cost")

    # Stack duals to (set_carriers, set_nodes, set_time_steps_operation)
    carrier_prices = duals.stack().dropna()
    carrier_prices.name = "carrier_price"

    flow_df = input_flow.reset_index()
    prices_df = carrier_prices.reset_index()

    merged = flow_df.merge(
        prices_df,
        left_on=["set_input_carriers", "set_nodes", "set_time_steps_operation"],
        right_on=["set_carriers", "set_nodes", "set_time_steps_operation"],
        how="inner",
    )
    merged["cost"] = merged["specific_input_flow_per_gw"] * merged["carrier_price"]

    annual_cost = (
        merged.groupby(["set_technologies", "set_input_carriers", "set_nodes"])["cost"]
        .sum()
    )

    if annual_cost.empty:
        print("\n--- Input carrier cost skipped: no matching carrier prices ---\n")
        return pd.Series(dtype=float, name="discounted_input_carrier_cost")

    group_index = annual_cost.index
    result = pd.Series(0.0, index=group_index, name="discounted_input_carrier_cost")

    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        capacity = float(capacity_addition_gw[tech])
        r = float(discount_rate.loc[(tech, node)])
        val = annual_cost.loc[(tech, carrier, node)]
        if pd.isna(val):
            continue
        total = sum(
            float(val) / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        result.loc[(tech, carrier, node)] = capacity * total

    return result

def get_tech_co2_cost_discounted(optimization_setup) -> pd.Series:
    """Discounted lifetime CO2 emission cost for the capacity addition returned by
    ``_capacity_addition``, for conversion technologies only.

    Mirrors ``get_variable_opex_discounted``, but multiplies the specific
    reference-carrier flow by ``carbon_intensity_technology`` [tCO2/MWh] and
    ``price_carbon_emissions`` [EUR/tCO2] of the base year.
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    op_level = "set_time_steps_operation"
    base_year = max(sets["set_time_steps_yearly"])

    # carbon intensity [tCO2/MWh] per (tech, node) — no time dimension in ZEN-Garden
    carbon_intensity = parameters.carbon_intensity_technology.to_series().dropna()
    if "set_location" in carbon_intensity.index.names:
        carbon_intensity.index = carbon_intensity.index.rename(
            {"set_location": "set_nodes"}
        )
    carbon_intensity = carbon_intensity[
        carbon_intensity.index.get_level_values("set_technologies").isin(techs)
    ]

    # CO2 price [EUR/tCO2] at base year
    price_co2 = parameters.price_carbon_emissions.to_series().dropna()
    price_co2_base = float(price_co2.loc[base_year])

    # Specific reference flow per GW [MWh/GW per time step]
    ref_flow = get_flow_reference_carrier(optimization_setup)
    if ref_flow.empty:
        return pd.Series(dtype=float, name="co2_cost_discounted")
    ref_flow = ref_flow.droplevel("reference_carrier")

    # [MWh/GW] * [tCO2/MWh] * [EUR/tCO2] = [EUR/GW] per time step, then sum per year
    ref_flow_df = ref_flow.rename("ref_flow").reset_index()
    ci_df = carbon_intensity.rename("ci").reset_index()
    product_df = ref_flow_df.merge(
        ci_df, on=["set_technologies", "set_nodes"], how="inner"
    )
    product_df["cost"] = (
        product_df["ref_flow"] * product_df["ci"] * price_co2_base
    )
    product_df["set_time_steps_yearly"] = product_df[op_level].map(op2year)

    annual_co2_cost = (
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
    result = pd.Series(0.0, index=idx, name="co2_cost_discounted")

    for tech in techs:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        cap = float(capacity_addition[tech])

        for node in nodes:
            r = float(discount_rate.loc[(tech, node)])
            try:
                co2_val = float(annual_co2_cost.loc[(tech, node, base_year)])
            except KeyError:
                continue  # tech has no emissions at this node

            total = sum(
                co2_val / (1 + r) ** offset
                for offset in range(delay, tech_lifetime + delay)
            )
            result.loc[(tech, node)] = cap * total

    return result

def get_carrier_co2_cost_discounted(optimization_setup) -> pd.Series:
    """Discounted lifetime CO2 cost from input carriers for the capacity addition
    returned by ``_capacity_addition``, for conversion technologies only.

    Analogous to ``calculate_input_carrier_cost``, but multiplies the specific
    input-carrier flow by ``carbon_intensity_carrier_import`` [tCO2/MWh] and the
    base-year ``price_carbon_emissions`` [EUR/tCO2]. Captures upstream/import
    emissions associated with the carriers a technology consumes — complementary
    to the direct technology emissions in ``get_tech_co2_cost_discounted``.
    """
    sets = optimization_setup.sets
    parameters = optimization_setup.parameters

    base_year = max(sets["set_time_steps_yearly"])

    # carbon intensity of carrier import [tCO2/MWh] per (carrier, node), base year
    carrier_ci = parameters.carbon_intensity_carrier_import.to_series().dropna()
    if "set_location" in carrier_ci.index.names:
        carrier_ci.index = carrier_ci.index.rename({"set_location": "set_nodes"})
    for year_level in ("set_time_steps_yearly", "year"):
        if year_level in carrier_ci.index.names:
            carrier_ci = carrier_ci.xs(base_year, level=year_level)
            break

    # CO2 price [EUR/tCO2] at base year
    price_co2 = parameters.price_carbon_emissions.to_series().dropna()
    price_co2_base = float(price_co2.loc[base_year])

    # specific input-carrier flow per GW [MWh/GW per time step]
    input_flow = get_flow_input_carrier(optimization_setup)
    if input_flow.empty:
        return pd.Series(dtype=float, name="carrier_co2_cost_discounted")

    # [MWh/GW] * [tCO2/MWh] * [EUR/tCO2] = [EUR/GW] per time step, then sum over time
    flow_df = input_flow.rename("flow").reset_index()
    ci_df = carrier_ci.rename("ci").reset_index().rename(
        columns={"set_carriers": "set_input_carriers"}
    )
    merged = flow_df.merge(
        ci_df, on=["set_input_carriers", "set_nodes"], how="inner"
    )
    if merged.empty:
        return pd.Series(dtype=float, name="carrier_co2_cost_discounted")

    merged["cost"] = merged["flow"] * merged["ci"] * price_co2_base

    annual_cost = (
        merged
        .groupby(["set_technologies", "set_input_carriers", "set_nodes"])["cost"]
        .sum()
    )

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition_gw = _capacity_addition(optimization_setup)

    group_index = annual_cost.index
    result = pd.Series(0.0, index=group_index, name="carrier_co2_cost_discounted")

    for tech, carrier, node in group_index:
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        capacity = float(capacity_addition_gw[tech])
        r = float(discount_rate.loc[(tech, node)])
        val = annual_cost.loc[(tech, carrier, node)]
        if pd.isna(val):
            continue
        total = sum(
            float(val) / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        result.loc[(tech, carrier, node)] = capacity * total

    return result


# subsidies
# Subsidies are positive cash flows that improve the profitability of a capacity
# addition. They are configured in the plugin config (``config["subsidies"]``) as
# a list of entries, each with the keys ``technology``, ``node``, ``type`` and
# ``amount`` (``remuneration`` entries additionally require ``carrier``). Four
# subsidy types are supported:
#   - ``capex``         : one-time relief in the decision year (NPV == annual).
#   - ``fixed_opex``    : annual lump-sum relief, recurring over the lifetime and
#                         discounted; scaled by the capacity addition.
#   - ``variable_opex`` : a per-MWh relief [money/MWh] on the reference-carrier
#                         flow (same logic as ``get_variable_opex_discounted``),
#                         recurring over the lifetime and discounted.
#   - ``remuneration``  : a fixed feed-in price [money/MWh] for the configured
#                         output ``carrier`` that replaces the market shadow
#                         price; the subsidy is the extra revenue
#                         (price - shadow_price) * production, discounted.
# Each function returns a DataFrame indexed by (set_conversion_technologies,
# set_nodes) with columns ``npv`` (discounted to the decision year, used in
# ``calculate_profitability``) and ``annual`` (undiscounted yearly value, written
# to the profitability CSV).
def _get_subsidies(optimization_setup, subsidy_type=None) -> list[dict]:
    """Validated subsidy entries from the plugin config.

    Reads ``config["subsidies"]`` from the investment_decisions plugin (lazy
    import to avoid a circular import) and returns a list of normalized dicts
    ``{technology, node, type, amount, carrier}`` (``carrier`` is ``None`` for
    all but ``remuneration`` entries). Entries that are malformed or that
    reference an unknown technology, node or subsidy type are dropped with a
    warning; ``remuneration`` entries are also dropped unless ``carrier`` names
    an output carrier of the technology. ``subsidy_type`` optionally restricts
    the result to a single type.
    """
    try:
        from zen_garden.plugins.investment_decisions.plugin import config
    except Exception:
        return []
    raw = config.get("subsidies", []) or []

    sets = optimization_setup.sets
    techs = set(sets["set_conversion_technologies"])
    nodes = set(sets["set_nodes"])
    valid_types = {"capex", "fixed_opex", "variable_opex", "remuneration"}

    out: list[dict] = []
    for entry in raw:
        try:
            tech = entry["technology"]
            node = entry["node"]
            stype = entry["type"]
            amount = float(entry["amount"])
        except (KeyError, TypeError, ValueError):
            logging.warning(f"Subsidy entry malformed, skipped: {entry!r}")
            continue
        if stype not in valid_types:
            logging.warning(f"Subsidy of unknown type {stype!r}, skipped: {entry!r}")
            continue
        if tech not in techs:
            logging.warning(
                f"Subsidy references unknown technology {tech!r}, skipped: {entry!r}"
            )
            continue
        if node not in nodes:
            logging.warning(
                f"Subsidy references unknown node {node!r}, skipped: {entry!r}"
            )
            continue

        # remuneration applies to one configured output carrier of the tech.
        carrier = entry.get("carrier")
        if stype == "remuneration":
            out_carriers = sets["set_output_carriers"][tech]  # tech already validated
            if carrier is None:
                logging.warning(
                    f"Remuneration subsidy requires an output 'carrier', "
                    f"skipped: {entry!r}"
                )
                continue
            if carrier not in out_carriers:
                logging.warning(
                    f"Remuneration subsidy carrier {carrier!r} is not an output "
                    f"carrier of {tech!r}, skipped: {entry!r}"
                )
                continue

        if subsidy_type is not None and stype != subsidy_type:
            continue
        out.append(
            {
                "technology": tech,
                "node": node,
                "type": stype,
                "amount": amount,
                "carrier": carrier,
            }
        )
    return out

def _subsidy_frame(records: list[dict]) -> pd.DataFrame:
    """Build the standard subsidy DataFrame from per-(tech, node) records.

    Returns a DataFrame indexed by (set_conversion_technologies, set_nodes) with
    columns ``npv`` and ``annual``. Multiple entries for the same (tech, node)
    are summed. An empty record list yields an empty, correctly-typed frame.
    """
    cols = ["npv", "annual"]
    if not records:
        idx = pd.MultiIndex.from_arrays(
            [[], []], names=["set_conversion_technologies", "set_nodes"]
        )
        return pd.DataFrame(0.0, index=idx, columns=cols)
    df = pd.DataFrame(records)
    # min_count=1 keeps an all-NaN group (e.g. the one-time CAPEX subsidy, which
    # has no recurring annual value) as NaN instead of collapsing it to 0.
    return df.groupby(["set_conversion_technologies", "set_nodes"])[cols].sum(
        min_count=1
    )

def _peer_avg_production(spec_prod: pd.Series, tech: str, carrier: str):
    """Average specific-production profile over peer nodes that produce.

    Used to estimate the production at a configured subsidy node that currently
    has no flow: returns the per-time-step mean of ``spec_prod`` across the nodes
    of ``(tech, carrier)`` whose annual production is positive, or ``None`` when
    no such peer node exists.
    """
    try:
        sub = spec_prod.loc[(tech, carrier)]  # indexed by (set_nodes, time step)
    except KeyError:
        return None
    node_sum = sub.groupby(level="set_nodes").sum()
    flow_nodes = node_sum.index[node_sum > 0]
    if len(flow_nodes) == 0:
        return None
    peer = sub[sub.index.get_level_values("set_nodes").isin(flow_nodes)]
    return peer.groupby(level="set_time_steps_operation").mean()

def get_capex_subsidy(optimization_setup) -> pd.DataFrame:
    """CAPEX subsidy: a one-time relief in the decision year, scaled per GW.

    For every ``capex`` subsidy entry the relief equals ``amount *
    capacity_addition`` and is paid once in the decision year (no discounting,
    mirroring the overnight CAPEX in ``get_capex``). As it is a one-time relief,
    it has no recurring yearly value, so ``annual`` is ``NaN``.
    """
    entries = _get_subsidies(optimization_setup, "capex")
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, amount = e["technology"], e["node"], e["amount"]
        value = amount * float(capacity_addition.get(tech, 0.0))
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": value,
                "annual": np.nan,
            }
        )
    return _subsidy_frame(records)

def get_fixed_opex_subsidy(optimization_setup) -> pd.DataFrame:
    """Fixed-OPEX subsidy: an annual lump-sum relief, scaled per GW.

    Each entry is an annual relief ``amount * capacity_addition`` that recurs
    over the technology lifetime and is discounted to the decision year with the
    same discount rate and investment delay as the OPEX cost components.
    """
    entries = _get_subsidies(optimization_setup, "fixed_opex")
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, amount = e["technology"], e["node"], e["amount"]
        if pd.isna(lifetime.get(tech, np.nan)):
            continue
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        annual = amount * float(capacity_addition.get(tech, 0.0))
        r = float(discount_rate.loc[(tech, node)])
        npv = sum(
            annual / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": npv,
                "annual": annual,
            }
        )
    return _subsidy_frame(records)

def get_variable_opex_subsidy(optimization_setup) -> pd.DataFrame:
    """Variable-OPEX subsidy: a per-MWh relief on the reference-carrier flow.

    Mirrors ``get_variable_opex_discounted``: the ``amount`` [money/MWh] is
    multiplied by the specific reference-carrier flow per GW (summed over the
    base year) and the capacity addition, then discounted over the lifetime with
    the same discount rate and investment delay as the OPEX cost components.

    If a configured node has no reference-carrier flow, the flow is estimated by
    the average flow over the nodes of the same technology that do have flow, so
    a subsidy can still be assigned (mirroring the profitability peer-fill).
    """
    entries = _get_subsidies(optimization_setup, "variable_opex")
    if not entries:
        return _subsidy_frame([])

    sets = optimization_setup.sets
    time_steps = optimization_setup.energy_system.time_steps
    op2year = pd.Series(time_steps.time_steps_operation2year)
    op_level = "set_time_steps_operation"
    base_year = max(sets["set_time_steps_yearly"])

    # specific reference-carrier flow per GW [MWh/GW per time step], summed per year
    ref_flow = get_flow_reference_carrier(optimization_setup)
    if ref_flow.empty:
        return _subsidy_frame([])
    ref_flow = ref_flow.droplevel("reference_carrier")
    ref_flow_df = ref_flow.rename("ref_flow").reset_index()
    ref_flow_df["set_time_steps_yearly"] = ref_flow_df[op_level].map(op2year)
    annual_ref_flow = (
        ref_flow_df
        .groupby(["set_technologies", "set_nodes", "set_time_steps_yearly"])["ref_flow"]
        .sum()
    )
    # base-year flow per (tech, node), used both directly and for the peer average
    base_flow = annual_ref_flow.xs(base_year, level="set_time_steps_yearly")

    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, amount = e["technology"], e["node"], e["amount"]
        if pd.isna(lifetime.get(tech, np.nan)):
            continue

        flow_val = float(base_flow.get((tech, node), 0.0))
        if not (flow_val > 0):
            # node has no reference-carrier flow -> estimate it with the average
            # flow over the nodes of the same tech that do have flow.
            tech_flows = (
                base_flow.loc[tech]
                if tech in base_flow.index.get_level_values("set_technologies")
                else pd.Series(dtype=float)
            )
            peers = tech_flows[tech_flows > 0]
            if len(peers) == 0:
                continue
            flow_val = float(peers.mean())

        annual = amount * flow_val * float(capacity_addition.get(tech, 0.0))
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        r = float(discount_rate.loc[(tech, node)])
        npv = sum(
            annual / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": npv,
                "annual": annual,
            }
        )
    return _subsidy_frame(records)

def get_remuneration_subsidy(optimization_setup) -> pd.DataFrame:
    """Remuneration subsidy: a fixed feed-in price replacing the shadow price.

    For every ``remuneration`` entry the ``amount`` is a fixed price
    [money/MWh] guaranteed for the technology's production of the configured
    output ``carrier`` at the node. The subsidy is the extra revenue over the
    market shadow price::

        annual = capacity_addition
                 * Σ_t specific_production[tech, carrier, node, t]
                   * (amount - shadow_price[carrier, node, t])

    for the single configured output carrier. The annual value is discounted
    over the lifetime exactly like ``calculate_revenue``.

    If a configured node has no production of the carrier, the production profile
    is estimated by the average over the peer nodes (same tech & carrier) that do
    produce; the market shadow price always stays node-specific.
    """
    entries = _get_subsidies(optimization_setup, "remuneration")
    if not entries:
        return _subsidy_frame([])

    spec_prod = get_specific_production(optimization_setup)
    prices = get_shadow_price(optimization_setup)
    discount_rate = get_discount_rate(optimization_setup)
    lifetime = get_lifetime(optimization_setup)
    investment_delay = _get_investment_delay(optimization_setup)
    capacity_addition = _capacity_addition(optimization_setup)

    records = []
    for e in entries:
        tech, node, price, carrier = (
            e["technology"], e["node"], e["amount"], e["carrier"]
        )
        if pd.isna(lifetime.get(tech, np.nan)):
            continue

        key = (tech, carrier, node)
        # specific production profile at the node; if the node does not produce
        # this carrier, fall back to the average profile over peer nodes that do.
        try:
            own = spec_prod.loc[key]
        except KeyError:
            own = None
        if own is not None and float(own.sum()) > 0:
            production = own
        else:
            production = _peer_avg_production(spec_prod, tech, carrier)
            if production is None:
                # no peer node produces this carrier -> cannot estimate
                continue

        try:
            market_price = prices.loc[key]
        except KeyError:
            market_price = pd.Series(0.0, index=production.index)
        diff = (price - market_price).reindex(production.index).fillna(price)
        annual = float((production * diff).sum()) * float(
            capacity_addition.get(tech, 0.0)
        )
        tech_lifetime = int(lifetime[tech])
        delay = int(investment_delay[tech])
        r = float(discount_rate.loc[(tech, node)])
        npv = sum(
            annual / (1 + r) ** offset
            for offset in range(delay, tech_lifetime + delay)
        )
        records.append(
            {
                "set_conversion_technologies": tech,
                "set_nodes": node,
                "npv": npv,
                "annual": annual,
            }
        )
    return _subsidy_frame(records)


# profitability calculation
def calculate_profitability(optimization_setup) -> pd.DataFrame:
    """Calculate profitability of the capacity addition as revenue minus costs.

    Costs include CAPEX, OPEX (fixed and variable), input-carrier cost and CO2
    emission cost (technology and carrier).  The same capacity addition,
    investment delay and discounting principles are applied as in the
    revenue and cost calculations.

    Configured subsidies (see the ``subsidies`` section) enter as additional
    positive components: their discounted NPV is added to the profitability and
    surfaced both as ``<type>_subsidy`` (NPV) and ``<type>_subsidy_annual``
    (undiscounted yearly value) columns.

    Zero-revenue (tech, node) pairs have their RAW profitability (before
    subsidies) replaced by the average raw profitability of revenue-positive
    nodes of the same technology (if any exist); the subsidies are added
    afterwards, so a filled node still keeps its own node-specific subsidy. Such
    entries are flagged in the ``values_changed`` column and their cost/revenue
    components (but not the subsidy components) are set to ``NaN``.

    Returns:
        ``pd.DataFrame`` indexed by (``set_conversion_technologies``,
        ``set_nodes``) with columns ``revenue``, ``capex``, ``fixed_opex``,
        ``variable_opex``, ``input_carrier_cost``, ``tech_co2_cost``,
        ``carrier_co2_cost``, the subsidy columns ``capex_subsidy``,
        ``fixed_opex_subsidy``, ``variable_opex_subsidy``,
        ``remuneration_subsidy`` (each with a ``_annual`` counterpart),
        ``profitability_raw`` (revenue minus costs, before subsidies),
        ``profitability`` (including subsidies), and ``values_changed`` (bool,
        True when the raw profitability was filled from peer nodes).
    """
    revenue = calculate_revenue(optimization_setup)
    capex = get_capex(optimization_setup)
    fixed_opex = get_fixed_opex_discounted(optimization_setup)
    variable_opex = get_variable_opex_discounted(optimization_setup)
    input_carrier_cost = calculate_input_carrier_cost(optimization_setup)
    tech_co2_cost = get_tech_co2_cost_discounted(optimization_setup)
    carrier_co2_cost = get_carrier_co2_cost_discounted(optimization_setup)

    # subsidies (positive cash flows): each frame carries a discounted ``npv``
    # (added to the profitability) and an undiscounted ``annual`` value (CSV).
    subsidy_frames = {
        "capex_subsidy": get_capex_subsidy(optimization_setup),
        "fixed_opex_subsidy": get_fixed_opex_subsidy(optimization_setup),
        "variable_opex_subsidy": get_variable_opex_subsidy(optimization_setup),
        "remuneration_subsidy": get_remuneration_subsidy(optimization_setup),
    }

    # revenue and input_carrier_cost are indexed by (set_technologies, *_carriers, set_nodes);
    # sum over carriers and rename to match the (set_conversion_technologies, set_nodes) index.
    def _by_tech_node(series):
        return (
            series
            .groupby(level=["set_technologies", "set_nodes"])
            .sum()
            .rename_axis(index={"set_technologies": "set_conversion_technologies"})
        )

    revenue_by_tech_node = _by_tech_node(revenue)
    input_carrier_cost_by_tech_node = _by_tech_node(input_carrier_cost)
    carrier_co2_cost_by_tech_node = _by_tech_node(carrier_co2_cost)

    profitability = revenue_by_tech_node.subtract(capex, fill_value=0)
    profitability = profitability.subtract(fixed_opex, fill_value=0)
    profitability = profitability.subtract(variable_opex, fill_value=0)
    profitability = profitability.subtract(input_carrier_cost_by_tech_node, fill_value=0)
    profitability = profitability.subtract(tech_co2_cost, fill_value=0)
    profitability = profitability.subtract(carrier_co2_cost_by_tech_node, fill_value=0)
    profitability.name = "profitability"

    values_changed = pd.Series(False, index=profitability.index, name="values_changed")

    # For (tech, node) pairs with 0 revenue, replace the RAW profitability (before
    # subsidies) with the average raw profitability of revenue-positive nodes of
    # the same technology. Filling happens before the subsidies are added, so only
    # the subsidy-free base is averaged; each filled node still receives its own
    # (flow-estimated) subsidy on top afterwards. If no peers exist, the value is
    # left unchanged.
    zero_rev_pairs = [
        idx for idx in profitability.index
        if revenue_by_tech_node.get(idx, 0.0) == 0.0
    ]
    for idx in zero_rev_pairs:
        tech, node = idx
        peer_profits = [
            float(profitability.loc[peer_idx])
            for peer_idx in profitability.index
            if peer_idx[0] == tech
            and peer_idx != idx
            and revenue_by_tech_node.get(peer_idx, 0.0) > 0.0
        ]
        if peer_profits:
            fill_val = float(np.mean(peer_profits))
            profitability.loc[idx] = fill_val
            values_changed.loc[idx] = True
            print(
                f"Profitability fill: ({tech}, {node}) has 0 revenue → "
                f"raw profitability replaced with avg of revenue-positive peers "
                f"= {fill_val:.2f}"
            )

    # raw profitability (revenue minus costs, after the peer-fill) before any
    # subsidies are added, surfaced as its own component for transparency.
    raw_profitability = profitability.copy()
    raw_profitability.name = "profitability_raw"

    # add the discounted subsidy NPVs (positive contributions) on top of the
    # possibly peer-filled raw profitability. Subsidy pairs are a subset of the
    # cost/revenue index, so the index is not extended.
    for frame in subsidy_frames.values():
        profitability = profitability.add(frame["npv"], fill_value=0)
    profitability.name = "profitability"

    components = pd.DataFrame(
        {
            "revenue": revenue_by_tech_node,
            "capex": capex,
            "fixed_opex": fixed_opex,
            "variable_opex": variable_opex,
            "input_carrier_cost": input_carrier_cost_by_tech_node,
            "tech_co2_cost": tech_co2_cost,
            "carrier_co2_cost": carrier_co2_cost_by_tech_node,
        }
    ).fillna(0.0)

    # subsidies as separate positive components: discounted ``npv`` (already part
    # of the profitability above) and the undiscounted ``annual`` value (CSV only).
    for name, frame in subsidy_frames.items():
        components[name] = frame["npv"].reindex(components.index, fill_value=0.0)
        components[f"{name}_annual"] = frame["annual"].reindex(
            components.index, fill_value=0.0
        )

    components["profitability_raw"] = raw_profitability.reindex(components.index)
    components["profitability"] = profitability.reindex(components.index)
    components["values_changed"] = values_changed.reindex(components.index, fill_value=False)

    # For manually filled entries the cost/revenue components are meaningless
    # (they came from a different node), so set them to NaN. The subsidy columns
    # are kept: they are estimated node-specifically (via peer-averaged flow) and
    # were added on top of the peer-averaged raw profitability.
    component_cols = [
        "revenue", "capex", "fixed_opex", "variable_opex",
        "input_carrier_cost", "tech_co2_cost", "carrier_co2_cost",
    ]
    components.loc[components["values_changed"], component_cols] = np.nan

    # one column per subsidy type (raw profitability sits after all costs and
    # before the subsidies, which then add up to the final profitability).
    subsidy_cols = list(subsidy_frames)
    subsidy_labels = {
        "capex_subsidy": "Capex Sub",
        "fixed_opex_subsidy": "FixOpx Sub",
        "variable_opex_subsidy": "VarOpx Sub",
        "remuneration_subsidy": "Remun Sub",
    }
    header = (
        f"{'Tech / Node':<45} {'Revenue':>12} {'CAPEX':>12} "
        f"{'Fixed OPEX':>12} {'Var OPEX':>12} {'Input Cost':>12} "
        f"{'Tech CO2':>12} {'Carrier CO2':>12} {'Raw Profit':>12} "
        + "".join(f"{subsidy_labels.get(c, c):>12} " for c in subsidy_cols)
        + f"{'Profit':>12} {'Changed':>9}"
    )
    print(
        f"\n--- Profitability of Capacity Additions ---\n"
        + header + "\n"
        + "-" * len(header)
    )
    for idx, row in components.iterrows():
        print(
            f"{str(idx):<45} {row['revenue']:>12.2f} {row['capex']:>12.2f} "
            f"{row['fixed_opex']:>12.2f} {row['variable_opex']:>12.2f} "
            f"{row['input_carrier_cost']:>12.2f} {row['tech_co2_cost']:>12.2f} "
            f"{row['carrier_co2_cost']:>12.2f} {row['profitability_raw']:>12.2f} "
            + "".join(f"{row[c]:>12.2f} " for c in subsidy_cols)
            + f"{row['profitability']:>12.2f} "
            f"{'*' if row['values_changed'] else '':>9}"
        )
    print()
    return components



# bias normalization
def get_npc_capex_coefficient(optimization_setup) -> pd.Series:
    """Per-GW coefficient that maps a capacity addition onto its CAPEX share of
    the base objective ``base = Σ_y NPC_y``, for **linear** conversion
    technologies only.

    Reconstructs the exact model chain that turns a capacity addition
    :math:`\\Delta S_{h,p,y'}` (built in the decision year ``y'``) into its
    contribution to the total net present cost, so that::

        capex_share_of_base[h, p] = coefficient[h, p] * capacity_addition[h, p]

    The decision year ``y'`` is the **first** year of the current foresight
    horizon (``set_time_steps_yearly[0]``), matching the variable that
    ``apply_profitability_bias_objective`` multiplies with the profitability
    bias. The coefficient is the partial derivative
    :math:`\\partial\\,\\text{base} / \\partial\\,\\Delta S_{h,p,y'}` and equals

    .. math::
        c_{h,p} = a_h \\; \\alpha_{h,p,y'}
                  \\sum_{y \\in \\mathcal{Y}\\,:\\,y' \\in \\mathcal{D}_h(y)}
                  \\phi_y

    with the three model ingredients:

    - **Annuity factor** :math:`a_h` (``constraint_cost_capex_yearly``)::

          a_h = ((1+r)^{lt} r) / ((1+r)^{lt} - 1)   (or 1/lt if r == 0)

      using the scalar discount rate ``r`` and the technology's
      ``depreciation_time`` ``lt``.
    - **Specific overnight CAPEX** :math:`\\alpha_{h,p,y'}` =
      ``capex_specific_conversion`` of the decision year
      (``constraint_linear_capex``; PWA technologies have no constant per-GW
      value and are returned as ``NaN``).
    - **Economic discount factor** :math:`\\phi_y`
      (``constraint_net_present_cost``), summed over every horizon year ``y``
      in whose depreciation range :math:`\\mathcal{D}_h(y)` the decision year
      ``y'`` still falls (``Technology.get_lifetime_range`` with
      ``use_depreciation_time=True``).

    Returns:
        ``pd.Series`` indexed by ``(set_conversion_technologies, set_nodes)``
        with the CAPEX coefficient in model money units per GW. PWA
        technologies are filled with ``NaN``.
    """
    from zen_garden.model.technology.technology import Technology
    from zen_garden.model.technology.conversion_technology import ConversionTechnology

    sets = optimization_setup.sets
    parameters = optimization_setup.parameters
    energy_system = optimization_setup.energy_system
    system = optimization_setup.system

    techs = list(sets["set_conversion_technologies"])
    nodes = list(sets["set_nodes"])
    years = list(sets["set_time_steps_yearly"])
    year_build = years[0]  # decision year: matches apply_profitability_bias_objective

    # scalar discount rate, exactly as used in the model constraints
    dr = float(np.asarray(parameters.discount_rate).item())
    depreciation_time = parameters.depreciation_time.to_series()

    # --- economic discount factor phi_y (mirror constraint_net_present_cost) ---
    entire_horizon = energy_system.set_time_steps_yearly_entire_horizon
    interval = system.interval_between_years
    discount_factor: dict[int, float] = {}
    for year in years:
        interval_between = 1 if year == entire_horizon[-1] else interval
        discount_factor[year] = sum(
            (1 / (1 + dr)) ** (interval * (year - year_build) + i)
            for i in range(0, interval_between)
        )

    # --- specific overnight CAPEX alpha of the decision year (linear only) ---
    capex_specific = parameters.capex_specific_conversion.to_series().dropna()
    for level_name in ("set_time_steps_yearly", "year"):
        if level_name in capex_specific.index.names:
            capex_specific = capex_specific.xs(year_build, level=level_name)
            break

    idx = pd.MultiIndex.from_product(
        [techs, nodes], names=["set_conversion_technologies", "set_nodes"]
    )
    result = pd.Series(np.nan, index=idx, name="npc_capex_coefficient")

    for tech in techs:
        # PWA technologies have no constant per-GW coefficient -> leave as NaN
        if optimization_setup.get_attribute_of_specific_element(
            ConversionTechnology, tech, "capex_is_pwa"
        ):
            continue

        lt = float(depreciation_time[tech])
        if dr != 0:
            annuity = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
        else:
            annuity = 1.0 / lt

        # discount factors of every horizon year whose depreciation range still
        # contains the decision year (i.e. the build still incurs annual capex)
        disc_sum = 0.0
        for year in years:
            deprange = Technology.get_lifetime_range(
                optimization_setup, tech, year, use_depreciation_time=True
            )
            if year_build in deprange:
                disc_sum += discount_factor[year]

        for node in nodes:
            if (tech, node) not in capex_specific.index:
                continue
            alpha = float(capex_specific.loc[(tech, node)])
            result.loc[(tech, node)] = annuity * alpha * disc_sum

    return result

def _get_bias_technologies(optimization_setup, output_carriers=None) -> set:
    """Conversion technologies eligible for the profitability bias.

    A technology qualifies when at least one of its output carriers is
    contained in ``output_carriers``. ``None`` or an empty value selects all
    conversion technologies (default behavior). A single carrier may be passed
    as a plain string.

    Returns:
        ``set`` of technology names. Empty (with a warning) when no conversion
        technology produces any of the configured carriers.
    """
    sets = optimization_setup.sets
    techs = list(sets["set_conversion_technologies"])
    if not output_carriers:
        return set(techs)
    if isinstance(output_carriers, str):
        output_carriers = [output_carriers]
    carriers = set(output_carriers)
    output_carriers_by_tech = sets["set_output_carriers"]
    selected = {
        tech for tech in techs
        if carriers.intersection(output_carriers_by_tech[tech])
    }
    if not selected:
        logging.warning(
            "Profitability bias: no conversion technology produces any of the "
            f"configured output carriers {sorted(carriers)}; bias has no effect."
        )
    return selected

def get_min_coefficient_profitability_ratio(
    optimization_setup, output_carriers=None, coefficient=None
) -> float:
    """Minimal ratio of NPC CAPEX coefficient to profitability over all
    (conversion technology, node) pairs.

    Per ``(set_conversion_technologies, set_nodes)`` computes::

        ratio = get_npc_capex_coefficient / optimization_setup.profitability

    aligning both series on their shared index, and returns the smallest finite
    ratio over the **profitable** pairs only (profitability > 0). Pairs whose
    coefficient is ``NaN`` (PWA technologies) or **zero** (pass-through
    technologies without CAPEX, e.g. ``oil_to_diesel_conversion``) or whose
    profitability is not strictly positive are excluded: a zero coefficient
    would yield a ratio of 0 and thereby cancel the entire bias term.

    When ``output_carriers`` is given, only technologies producing at least one
    of these carriers are considered (see ``_get_bias_technologies``); ``None``
    keeps all conversion technologies.

    The profitability is read from ``optimization_setup.profitability`` (set by
    ``after_optimization_event`` after each solve). When the attribute is absent
    (e.g. the first rolling-horizon step), ``NaN`` is returned.

    Args:
        optimization_setup: the optimization setup of the current step.
        output_carriers: optional carrier restriction (see above).
        coefficient: pre-computed result of ``get_npc_capex_coefficient``; when
            provided the function is not called again.

    Returns:
        ``float`` minimal ratio, or ``NaN`` if no profitable pair remains.
    """
    profitability = getattr(optimization_setup, "profitability", None)
    if profitability is None:
        # no profitability signal yet -> no ratio available
        print("\n--- Coefficient / profitability ratio: no profitability signal yet ---\n")
        return float("nan")

    if coefficient is None:
        coefficient = get_npc_capex_coefficient(optimization_setup)
    # exclude zero-CAPEX (pass-through) technologies: their ratio of 0 would
    # collapse the bias term to plain net present cost for every bias weight
    coefficient = coefficient[coefficient != 0]

    # restrict to technologies with a configured output carrier
    bias_techs = _get_bias_technologies(optimization_setup, output_carriers)
    profitability = profitability[
        profitability.index.get_level_values("set_conversion_technologies").isin(bias_techs)
    ]

    # restrict to profitable pairs only
    profitability = profitability[profitability > 0]

    ratio = coefficient.div(profitability)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio.name = "coefficient_profitability_ratio"

    if ratio.empty:
        print("\n--- Coefficient / profitability ratio: no profitable (tech, node) pairs ---\n")
        return float("nan")

    min_idx = ratio.idxmin()
    min_val = float(ratio.loc[min_idx])

    print(
        f"\n--- CAPEX-coefficient / profitability ratios (profitable pairs) ---\n"
        f"{'Tech / Node':<45} {'Ratio':>12}\n"
        + "-" * 58
    )
    for idx, val in ratio.sort_values().items():
        marker = "  <- min" if idx == min_idx else ""
        print(f"{str(idx):<45} {val:>12.4e}{marker}")
    print()

    return min_val




# objective modification
def apply_profitability_bias_objective(
    optimization_setup, weight: float = 0.5, output_carriers=None
) -> None:
    """Replace the model objective with total NPC penalized by a profitability bias.

    Modifies the already-constructed objective in place using the
    ``remove_objective`` / ``add_objective`` pattern:

    .. math::
        J = \\sum_{y\\in\\mathcal{Y}} NPC_y
            - \\rho_{\\min}\\, w \\sum_{t,n} \\pi^{\\text{last}}_{t,n} \\cdot
              \\text{capacity\\_addition}_{t,\\text{power},n,y_{now}}

    ``\\pi^{\\text{last}}_{t,n}`` is read from ``optimization_setup.profitability``
    (set by ``after_optimization_event`` after each solve). When the attribute is
    absent (e.g. the first rolling-horizon step), the objective is left unchanged
    and collapses to the plain total net present cost.

    ``\\rho_{\\min}`` is the minimal CAPEX-coefficient / profitability ratio over
    the profitable (tech, node) pairs, from
    ``get_min_coefficient_profitability_ratio``. If it is not finite or not
    strictly positive (no profitable pair with a positive CAPEX coefficient),
    the bias term is dropped and the objective collapses to the plain total net
    present cost.

    Technologies with a CAPEX coefficient of exactly 0 (pass-through
    converters) receive no bias: they are excluded from ``\\rho_{\\min}`` and
    their bias coefficient is set to 0, since a bias on a technology without
    capital cost would subsidize otherwise free capacity additions.

    When ``output_carriers`` is given, the bias is restricted to technologies
    producing at least one of these carriers: every other technology receives a
    coefficient of 0 and ``\\rho_{\\min}`` is likewise computed over the
    eligible technologies only. ``None`` (default) applies the bias to all
    conversion technologies.

    Args:
        optimization_setup: The optimization setup holding the constructed model.
        weight: Bias weight ``w`` applied to the profitability term.
        output_carriers: Optional list of output carrier names (or a single
            name) limiting which technologies are biased.
    """
    profit = getattr(optimization_setup, "profitability", None)
    if profit is None:
        # no profitability signal yet -> keep the base total-cost objective
        return

    # zero the coefficient of technologies without a configured output carrier
    bias_techs = _get_bias_technologies(optimization_setup, output_carriers)
    profit = profit.where(
        profit.index.get_level_values("set_conversion_technologies").isin(bias_techs),
        0.0,
    )

    # zero the coefficient of (tech, node) pairs without CAPEX (pass-through
    # converters such as oil_to_diesel_conversion): biasing them would
    # subsidize capacity additions that cost nothing
    capex_coefficient = get_npc_capex_coefficient(optimization_setup)
    zero_capex = capex_coefficient.index[capex_coefficient == 0]
    if len(zero_capex) > 0:
        profit = profit.where(~profit.index.isin(zero_capex), 0.0)
        print(
            f"Profitability bias: excluded {len(zero_capex)} (tech, node) pairs "
            f"with zero CAPEX coefficient from the bias "
            f"({sorted(set(zero_capex.get_level_values(0)))})"
        )

    ratio_min = get_min_coefficient_profitability_ratio(
        optimization_setup, output_carriers=output_carriers,
        coefficient=capex_coefficient,
    )
    if not np.isfinite(ratio_min) or ratio_min <= 0:
        # no profitable (tech, node) pair with positive CAPEX coefficient
        # -> keep the base total-cost objective
        logging.warning(
            "apply_profitability_bias_objective: minimal coefficient/profitability "
            f"ratio ({ratio_min}) is not finite and positive; keeping plain net "
            "present cost objective."
        )
        return

    model = optimization_setup.model
    base = model.variables["net_present_cost"].sum("set_time_steps_yearly")
    year_now = optimization_setup.energy_system.set_time_steps_yearly[0]
    cap_add = model.variables["capacity_addition"].sel(
        set_capacity_types="power", set_time_steps_yearly=year_now,
    )
    coef = (
        profit
        .rename_axis(index={
            "set_conversion_technologies": "set_technologies",
            "set_nodes": "set_location",
        })
        .to_xarray()
        .reindex(
            set_technologies=cap_add.coords["set_technologies"],
            set_location=cap_add.coords["set_location"],
            fill_value=0.0,
        )
    )
    coef_series = coef.to_series()
    print(
        f"\n--- Profitability bias coefficients "
        f"(ratio_min={ratio_min:.4e}, weight={weight}) ---\n"
        f"{'Tech / Node':<45} {'Coef':>14} {'Effective':>14}\n"
        + "-" * 75
    )
    for idx, val in coef_series.items():
        print(f"{str(idx):<45} {val:>14.4f} {ratio_min * weight * val:>14.4e}")
    print()

    objective = base - ratio_min * (weight * coef * cap_add).sum()

    optimization_setup.model.remove_objective()
    optimization_setup.model.add_objective(
        objective, sense=optimization_setup.analysis.sense
    )



#visualization and analysis output
def save_profitability_components(
    optimization_setup,
    output_dir=None,
    components_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Save all profitability components of the current step as a CSV.

    Calls ``calculate_profitability`` to obtain every component (revenue, CAPEX,
    fixed/variable OPEX, input-carrier cost, CO2 costs, net profitability, and
    the ``values_changed`` flag) and appends them — tagged with the decision year
    and carrier metadata — to ``profitability_components.csv`` in the timestamped
    per-run folder.

    Args:
        optimization_setup: solved optimization setup of the current step.
        output_dir: root visualization directory; defaults to the dataset's
            ``visualization`` folder derived from ``analysis.dataset``.
        components_df: pre-computed result of ``calculate_profitability``; when
            provided the function is not called again, avoiding double computation.

    Returns:
        ``pd.DataFrame`` with one row per (tech, node) and the components written
        for this step.
    """
    year = int(max(optimization_setup.sets["set_time_steps_yearly"]))

    if components_df is None:
        components_df = calculate_profitability(optimization_setup)

    df = components_df.copy()
    df.index = df.index.set_names(["set_conversion_technologies", "set_nodes"])
    df = df.reset_index()

    sets = optimization_setup.sets

    def _carriers(mapping, tech):
        try:
            vals = mapping[tech]
        except (KeyError, TypeError):
            return ""
        return ", ".join(str(c) for c in vals)

    techs = df["set_conversion_technologies"]
    df.insert(
        1, "output_carrier",
        techs.map(lambda t: _carriers(sets["set_output_carriers"], t)),
    )
    df.insert(
        2, "input_carrier",
        techs.map(lambda t: _carriers(sets["set_input_carriers"], t)),
    )
    df.insert(
        3, "reference_carrier",
        techs.map(lambda t: _carriers(sets["set_reference_carriers"], t)),
    )
    df.insert(0, "decision_year", year)

    if output_dir is None:
        out = _get_run_output_dir(optimization_setup)
    else:
        from pathlib import Path

        out = Path(output_dir) / _get_run_timestamp()
        out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "profitability_components.csv"
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)
    print(
        f"Saved profitability components ({len(df)} rows, decision year {year}) "
        f"to {csv_path}"
    )
    return df

def visualization(
    optimization_setup,
    output_dir: str | None = None,
) -> None:
    """Generate and save investment-decision visualizations.

    Saves two plot types **per output carrier** to ``output_dir``: one pair of
    charts for every output carrier, listing all (technology, node) pairs whose
    technology produces that carrier. A technology with several output carriers
    therefore appears in several carrier charts (with the same profitability,
    since CAPEX/OPEX/CO2 costs are not carrier-specific). Each file name is
    suffixed with the carrier and the current calendar year so runs for
    different carriers / optimization periods do not overwrite each other:
    - ``profitability_breakdown_<carrier>_<year>.png``: revenue vs. stacked costs + net-profit marker per tech/node
    - ``profitability_net_<carrier>_<year>.png``: net profit bar chart, color-coded profitable / loss

    All plots of one program run are written into a timestamped subfolder of
    the dataset's ``visualization`` directory (shared across every optimization
    step of that run). ``output_dir`` overrides that root if given.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path

    # per-run timestamped subfolder, identical for all steps of one program run
    if output_dir is None:
        out = _get_run_output_dir(optimization_setup)
    else:
        out = Path(output_dir) / _get_run_timestamp()
        out.mkdir(parents=True, exist_ok=True)

    year = max(optimization_setup.sets["set_time_steps_yearly"])

    # --- net profitability via central function ---
    pro = calculate_profitability(optimization_setup)["profitability"]

    idx = pro.index
    if idx.empty:
        print("visualization: keine Profitabilitätsdaten vorhanden, Diagramme werden übersprungen.")
        return

    # --- individual cost/revenue components for the breakdown chart ---
    def _by_tech_node(series):
        return (
            series
            .groupby(level=["set_technologies", "set_nodes"])
            .sum()
            .rename_axis(index={"set_technologies": "set_conversion_technologies"})
            .reindex(idx, fill_value=0)
        )

    rev = _by_tech_node(calculate_revenue(optimization_setup))
    cap = get_capex(optimization_setup).reindex(idx, fill_value=0)
    fop = get_fixed_opex_discounted(optimization_setup).reindex(idx, fill_value=0)
    vop = get_variable_opex_discounted(optimization_setup).reindex(idx, fill_value=0)
    icc = _by_tech_node(calculate_input_carrier_cost(optimization_setup))
    tco2 = get_tech_co2_cost_discounted(optimization_setup).reindex(idx, fill_value=0)
    cco2 = _by_tech_node(get_carrier_co2_cost_discounted(optimization_setup))

    # one pair of charts per output carrier; all (tech, node) pairs whose
    # technology produces that carrier appear on the x-axis.
    output_carriers_by_tech = optimization_setup.sets["set_output_carriers"]
    technologies = list(dict.fromkeys(idx.get_level_values("set_conversion_technologies")))

    carrier_to_techs: dict[str, list[str]] = {}
    for tech in technologies:
        for carrier in output_carriers_by_tech[tech]:
            carrier_to_techs.setdefault(carrier, []).append(tech)

    if not carrier_to_techs:
        print("visualization: keine Output-Carrier gefunden, Diagramme werden übersprungen.")
        return

    for carrier, carrier_techs in carrier_to_techs.items():
        carrier_techs_set = set(carrier_techs)
        # rows of the shared index whose technology produces this carrier
        mask = idx.get_level_values("set_conversion_technologies").isin(carrier_techs_set)
        if not mask.any():
            continue

        # slice every component down to the current carrier's (tech, node) pairs
        rev_t = rev[mask]
        cap_t = cap[mask]
        fop_t = fop[mask]
        vop_t = vop[mask]
        icc_t = icc[mask]
        tco2_t = tco2[mask]
        cco2_t = cco2[mask]
        pro_t = pro[mask]

        # technologies/nodes with zero revenue have no existing capacity for this
        # carrier; drop them from the plots and list them as text annotation.
        nonzero_mask = rev_t.values != 0
        zero_pairs = [f"{t} / {n}" for (t, n), keep in zip(rev_t.index, nonzero_mask) if not keep]
        rev_t = rev_t[nonzero_mask]
        cap_t = cap_t[nonzero_mask]
        fop_t = fop_t[nonzero_mask]
        vop_t = vop_t[nonzero_mask]
        icc_t = icc_t[nonzero_mask]
        tco2_t = tco2_t[nonzero_mask]
        cco2_t = cco2_t[nonzero_mask]
        pro_t = pro_t[nonzero_mask]

        no_cap_text = (
            "No existing capacity: " + ", ".join(zero_pairs) if zero_pairs else ""
        )

        if pro_t.empty:
            # nothing left to plot, but still surface the skipped entries
            print(
                f"visualization: carrier '{carrier}' – alle Technologien ohne Revenue. "
                f"{no_cap_text}"
            )
            continue

        labels = [f"{t}\n{n}" for t, n in pro_t.index]
        x = list(range(len(labels)))
        fig_width = max(7, len(labels) * 1.6)

        # -------------------------------------------------------------- #
        # Plot 1 – Revenue vs. Costs Breakdown + Net Profit Marker        #
        # -------------------------------------------------------------- #
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        w = 0.45

        ax.bar(x, rev_t.values, w, label="Revenue", color="#2ecc71", zorder=3)
        ax.bar(x, -cap_t.values, w, label="CAPEX", color="#e74c3c", zorder=3)
        ax.bar(x, -fop_t.values, w, label="Fixed OPEX", color="#e67e22",
               bottom=-cap_t.values, zorder=3)
        ax.bar(x, -vop_t.values, w, label="Var. OPEX", color="#f39c12",
               bottom=(-cap_t - fop_t).values, zorder=3)
        ax.bar(x, -icc_t.values, w, label="Input Cost", color="#9b59b6",
               bottom=(-cap_t - fop_t - vop_t).values, zorder=3)
        ax.bar(x, -tco2_t.values, w, label="Tech CO2", color="#7f8c8d",
               bottom=(-cap_t - fop_t - vop_t - icc_t).values, zorder=3)
        ax.bar(x, -cco2_t.values, w, label="Carrier CO2", color="#34495e",
               bottom=(-cap_t - fop_t - vop_t - icc_t - tco2_t).values, zorder=3)
        ax.plot(x, pro_t.values, "D", color="#2c3e50", markersize=9,
                label="Net Profit", zorder=4, clip_on=False)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Technology / Node")
        ax.set_ylabel("Discounted value [model money units]")
        ax.set_title(f"Investment Profitability Breakdown – output carrier '{carrier}' ({year})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if no_cap_text:
            fig.text(0.5, 0.01, no_cap_text, ha="center", va="bottom",
                     fontsize=8, style="italic", wrap=True)
            plt.tight_layout(rect=(0, 0.05, 1, 1))
        else:
            plt.tight_layout()
        p1 = out / f"profitability_breakdown_{carrier}_{year}.png"
        fig.savefig(p1, dpi=150)
        plt.close(fig)
        print(f"Saved: {p1}")

        # -------------------------------------------------------------- #
        # Plot 2 – Net Profitability Bar Chart                            #
        # -------------------------------------------------------------- #
        bar_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in pro_t.values]
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        bars = ax.bar(x, pro_t.values, color=bar_colors, edgecolor="white", zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

        value_range = max(abs(pro_t.values)) if len(pro_t) else 1
        bar_offset = (value_range or 1) * 0.02
        for bar, val in zip(bars, pro_t.values):
            y = val + bar_offset if val >= 0 else val - bar_offset
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:,.0f}", ha="center", va=va, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Technology / Node")
        ax.set_ylabel("Net Discounted Profit [model money units]")
        ax.set_title(f"Net Investment Profitability – output carrier '{carrier}' ({year})")
        handles = [
            mpatches.Patch(color="#27ae60", label="Profitable"),
            mpatches.Patch(color="#c0392b", label="Loss"),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if no_cap_text:
            fig.text(0.5, 0.01, no_cap_text, ha="center", va="bottom",
                     fontsize=8, style="italic", wrap=True)
            plt.tight_layout(rect=(0, 0.05, 1, 1))
        else:
            plt.tight_layout()
        p2 = out / f"profitability_net_{carrier}_{year}.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"Saved: {p2}")

