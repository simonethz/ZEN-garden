"""Investment decisions plugin.

Subscribes to optimization events to extract and print investor-relevant
signals (e.g. nodal shadow prices) from the solved model.
The config dictionary will be filled by plugins.loader.register_plugins()
"""

import logging

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.investment_decisions.investment_decisions import (
    extract_average_shadow_prices, calculate_revenue, get_specific_capex
)

config = {}


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    
    # print average shadow prices
    average_prices = extract_average_shadow_prices(optimization_setup)
    if average_prices is not None:
        logging.info(f"\n--- Average Shadow Prices ---\n{average_prices}\n")
    else:
        logging.warning("No average shadow prices available to print.")

    # print revenue calculation
    revenue = calculate_revenue(optimization_setup)
    if revenue is not None:
        logging.info(f"\n--- Revenue ---\n{revenue}\n")
    else:
        logging.warning("No revenue available to print.")

    # call capex
    get_specific_capex(optimization_setup)