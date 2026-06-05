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
    apply_profitability_bias_objective, calculate_input_carrier_cost, calculate_profitability, extract_average_shadow_prices, calculate_revenue, get_capex, get_fixed_opex_discounted, get_flow_reference_carrier, get_min_coefficient_profitability_ratio, get_variable_opex_discounted, visualization
)

config = {}

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding="utf-8")
        
        # Optional: Einen Zeitstempel in die Datei schreiben, sobald das Event auslöst
        jetzt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"\n=== Event 'after_optimization' getriggert am {jetzt} ===\n")

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
        yield # Hier läuft der Code deines with-Blocks
    finally:
        sys.stdout = original_stdout # Standardausgabe wiederherstellen
        logger.log_file.close()      # Datei sicher schließen


@EventPublisher.register(Event.event_before_solve)
def before_solve_event(optimization_setup):
    """Bias the freshly constructed objective with the previous step's profitability.

    Only active when ``profitability_bias_enabled`` is set in the plugin config
    (analogous to ``bias_weight``). On the first rolling-horizon step (no
    profitability available yet) the objective is left untouched.
    """
    if not config.get("profitability_bias_enabled", False):
        return
    apply_profitability_bias_objective(
        optimization_setup, weight=config.get("bias_weight", 1.0)
    )


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    with log_block(r"D:\Students\ssambale_jwiegner\ZEN-garden\investment_plugin_output.txt"):
    
        print("\n--- Calculating and printing discounted fixed OPEX ---")
       
        extract_average_shadow_prices(optimization_setup)        

        # publish profitability for use as bias in the next period's objective
        # (consumed by before_solve_event -> apply_profitability_bias_objective).
        optimization_setup.profitability = calculate_profitability(optimization_setup)
    visualization(optimization_setup)
