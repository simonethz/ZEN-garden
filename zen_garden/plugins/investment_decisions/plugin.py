"""Investment decisions plugin.

Subscribes to optimization events to extract and print investor-relevant
signals (e.g. nodal shadow prices) from the solved model.
The config dictionary will be filled by plugins.loader.register_plugins()
"""

import logging
import sys
import datetime
import contextlib

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.investment_decisions.investment_decisions import (
    apply_profitability_bias_objective, calculate_input_carrier_cost, calculate_profitability, extract_average_shadow_prices, calculate_revenue, get_capex, get_fixed_opex_discounted, get_flow_reference_carrier, get_min_coefficient_profitability_ratio, get_variable_opex_discounted, save_profitability_components, visualization
)

config = {}

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding="utf-8")
        
        # write a timestamp to the file when the event fires
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"\n=== Event 'after_optimization' triggered at {now} ===\n")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

@contextlib.contextmanager
def log_block(filename):
    logger = DualLogger(filename)
    original_stdout = sys.stdout
    sys.stdout = logger
    try:
        yield  # the body of the with-block runs here
    finally:
        sys.stdout = original_stdout  # restore standard output
        logger.log_file.close()       # close the file safely


@EventPublisher.register(Event.event_before_solve)
def before_solve_event(optimization_setup):
    """Bias the freshly constructed objective with the previous step's profitability.

    Only active when ``profitability_bias_enabled`` is set in the plugin config
    (analogous to ``bias_weight``). On the first rolling-horizon step (no
    profitability available yet) the objective is left untouched.

    ``bias_output_carriers`` (optional, list of carrier names) restricts the
    bias to technologies producing at least one of these carriers; all other
    technologies receive a coefficient of 0. When absent or empty, all
    conversion technologies are biased.
    """
    with log_block(r"D:\Students\ssambale_jwiegner\ZEN-garden\investment_plugin_output.txt"):
        print("\n--- Applying profitability bias to objective ---")
        if not config.get("profitability_bias_enabled", False):
            return
        apply_profitability_bias_objective(
            optimization_setup,
            weight=config.get("bias_weight", 0.5),
            output_carriers=config.get("bias_output_carriers", []),
        )


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    with log_block(r"D:\Students\ssambale_jwiegner\ZEN-garden\investment_plugin_output.txt"):
    
        print("\n--- Calculating and printing discounted fixed OPEX ---")
       
        extract_average_shadow_prices(optimization_setup)        

        # compute once; reuse the result for the bias signal and the CSV export.
        profitability_components = calculate_profitability(optimization_setup)

        # publish profitability Series for use as bias in the next period's
        # objective (consumed by before_solve_event -> apply_profitability_bias_objective).
        optimization_setup.profitability = profitability_components["profitability"]

        # persist all profitability components of this step to a per-run CSV so
        # they can be reused later (e.g. by run_and_visualize) across all steps.
        save_profitability_components(optimization_setup, components_df=profitability_components)
    visualization(optimization_setup)
