"""Investment decisions plugin.

Subscribes to optimization events to extract and print investor-relevant
signals (e.g. nodal shadow prices) from the solved model.
The config dictionary will be filled by plugins.loader.register_plugins()
"""

import logging

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.investment_decisions.investment_decisions import (
    calculate_input_carrier_cost, calculate_profitability, extract_average_shadow_prices, calculate_revenue, get_capex, get_fixed_opex_discounted, get_flow_reference_carrier, get_variable_opex_discounted, visualization
)

config = {}


@EventPublisher.register(Event.event_after_optimization)
def after_optimization_event(optimization_setup):
    
    
    #call opex
    print("\n--- Calculating and printing discounted fixed OPEX ---")
    #get_fixed_opex_discounted(optimization_setup)
    #get_flow_reference_carrier(optimization_setup)
    #get_variable_opex_discounted(optimization_setup)
    #calculate_input_carrier_cost(optimization_setup)
    #calculate_profitability(optimization_setup)
    visualization(optimization_setup)
