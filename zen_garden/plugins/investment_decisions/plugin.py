"""Investment decisions plugin.

Subscribes to optimization events to extract and print investor-relevant
signals (e.g. nodal shadow prices) from the solved model.
The config dictionary will be filled by plugins.loader.register_plugins()
"""

import logging

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.investment_decisions.investment_decisions import (
    calculate_profitability, extract_average_shadow_prices, calculate_revenue, get_capex, get_fixed_opex_discounted, get_flow_reference_carrier, get_variable_opex_discounted, visualization
)

config = {}


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    
    # print average shadow prices
    print("\n--- Extracting and printing average shadow prices ---")
    average_prices = extract_average_shadow_prices(optimization_setup)
    if average_prices is not None:
        logging.info(f"\n--- Average Shadow Prices ---\n{average_prices}\n")
    else:
        logging.warning("No average shadow prices available to print.")

    # print revenue calculation
    print("\n--- Calculating and printing revenue ---")
    revenue = calculate_revenue(optimization_setup)
    if revenue is not None:
        logging.info(f"\n--- Revenue ---\n{revenue}\n")
    else:
        logging.warning("No revenue available to print.")

    # call capex
    print("\n--- Calculating and printing CAPEX ---")
    get_capex(optimization_setup)

    #call opex
    print("\n--- Calculating and printing discounted fixed OPEX ---")
    #get_fixed_opex_discounted(optimization_setup)
    #get_flow_reference_carrier(optimization_setup)
    #get_variable_opex_discounted(optimization_setup)
    calculate_profitability(optimization_setup)
    visualization(optimization_setup)
