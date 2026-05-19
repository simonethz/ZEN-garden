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
    calculate_input_carrier_cost, calculate_profitability, extract_average_shadow_prices, calculate_revenue, get_capex, get_fixed_opex_discounted, get_flow_reference_carrier, get_variable_opex_discounted, visualization
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


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    with log_block(r"D:\Students\ssambale_jwiegner\ZEN-garden\investment_plugin_output.txt"):
    
        #call opex
        print("\n--- Calculating and printing discounted fixed OPEX ---")
        #get_fixed_opex_discounted(optimization_setup)
        #get_flow_reference_carrier(optimization_setup)
        #get_variable_opex_discounted(optimization_setup)
        #calculate_input_carrier_cost(optimization_setup)
        #calculate_profitability(optimization_setup)
        visualization(optimization_setup)
