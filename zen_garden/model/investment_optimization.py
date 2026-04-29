import logging
import pandas as pd

def extract_and_print_market_prices(optimization_setup):
    """
    Extracts the shadow prices (dual variables) of the nodal energy balance constraint
    and returns the yearly mean per carrier and node.

    The yearly mean is the duration-weighted average of the per-hour shadow prices over
    the optimized period, i.e. sum(dual_c) / sum(duration_c). This compensates for the
    fact that the raw dual of an aggregated time step scales with the number of physical
    hours that cluster represents.
    """
    constraint_name = "constraint_nodal_energy_balance"
    try:
        model = optimization_setup.model
        if not (hasattr(model, "constraints") and constraint_name in model.constraints):
            logging.warning(f"Constraint '{constraint_name}' not found in the model. Cannot extract shadow prices.")
            return None

        duals = model.constraints[constraint_name].dual
        if duals is None:
            logging.warning(f"Duals for '{constraint_name}' are currently None (solver might not provide them for this run).")
            return None

        logging.info(f"\n--- Yearly Mean Shadow Prices (Market Prices) for '{constraint_name}' ---")

        # 1. Convert xarray to dataframe and unstack over operation time steps
        df_duals = duals.to_dataframe(name="dual").unstack("set_time_steps_operation")
        df_duals.columns = df_duals.columns.get_level_values("set_time_steps_operation")

        # 2. Duration-weighted yearly mean per (carrier, node):
        #     mean_per_hour = sum_c(dual_c) / sum_c(duration_c)
        # The raw dual of an aggregated time step already equals duration * per-hour price,
        # so summing the raw duals and dividing by total duration gives the yearly average.
        durations = pd.Series(
            optimization_setup.energy_system.time_steps.time_steps_operation_duration
        ).reindex(df_duals.columns)
        total_duration = durations.sum()
        yearly_mean = df_duals.sum(axis=1) / total_duration
        yearly_mean.name = "yearly_mean_shadow_price"

        print(f"\nYeae duals in investmennt_optimization.pyrly mean shadow price per (carrier, node) for {constraint_name} "
              f"(over {int(total_duration)} h):")
        print(yearly_mean)

        return yearly_mean
    except Exception as e:
        logging.error(f"Failed to extract shadow prices for {constraint_name}: {e}")
        return None

