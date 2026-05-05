"""Investment decisions plugin.

Subscribes to optimization events to extract and print investor-relevant
signals (e.g. nodal shadow prices) from the solved model.
The config dictionary will be filled by plugins.loader.register_plugins()
"""

import logging

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.investment_decisions.investment_decisions import (
    extract_average_shadow_prices,
)

config = {}


@EventPublisher.register(Event.event_after_optimization)
def print_average_shadow_prices(optimization_setup):
    """Print the average shadow prices of nodal energy balance duals.

    Args:
        optimization_setup: A solved ``OptimizationSetup`` whose solver was run
            with ``save_duals = True``.
    """
    average_prices = extract_average_shadow_prices(optimization_setup)
    if average_prices is not None:
        logging.info(f"\n--- Average Shadow Prices ---\n{average_prices}\n")
    else:
        logging.warning("No average shadow prices available to print.")