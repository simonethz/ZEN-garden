import logging
import os
from pathlib import Path

import cloudpickle
import pandas as pd

from zen_garden.model.technology.technology import Technology
from zen_garden.model.technology.storage_technology import StorageTechnology

DEFAULT_OPTIMIZATION_SETUP_PICKLE = (
    Path(__file__).resolve().parents[2] / "optimization_setup.pkl"
)


def save_optimization_setup(optimization_setup, path: os.PathLike | str | None = None) -> Path:
    """Pickles a live OptimizationSetup to disk so it can be reloaded for testing
    `investment_optimization` helpers without re-running the full pipeline.

    Uses ``cloudpickle`` because OptimizationSetup holds a Linopy model with xarray
    objects and bound methods that stdlib ``pickle`` cannot always serialize.

    :param optimization_setup: live OptimizationSetup (ideally after solve)
    :param path: target file; defaults to ``<repo>/optimization_setup.pkl``
    :return: resolved Path the object was written to
    """
    target = Path(path) if path is not None else DEFAULT_OPTIMIZATION_SETUP_PICKLE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        cloudpickle.dump(optimization_setup, f)
    logging.info(f"Saved optimization_setup to {target}")
    return target


def load_optimization_setup(path: os.PathLike | str | None = None):
    """Loads an OptimizationSetup previously written by :func:`save_optimization_setup`.

    :param path: source file; defaults to ``<repo>/optimization_setup.pkl``
    :return: the unpickled OptimizationSetup instance
    """
    source = Path(path) if path is not None else DEFAULT_OPTIMIZATION_SETUP_PICKLE
    with open(source, "rb") as f:
        return cloudpickle.load(f)


def compute_capex_and_opex_of_capacity(
    optimization_setup,
    technology_name: str,
    location: str,
    year: int,
    capacity: float,
    storage_energy: bool = False,
):
    """Computes the (annualized) CAPEX and the annual fixed OPEX of adding
    `capacity` of `technology_name` at `location` in `year`.

    Reuses the existing per-technology helper
    `Technology.calculate_capex_of_single_capacity` for CAPEX (defined in
    technology/conversion_technology.py:137, storage_technology.py:151,
    transport_technology.py:185). Note: for the linear (non-PWA) case the helper
    multiplies by the first-year specific capex; for PWA it interpolates the
    full curve.

    Fixed OPEX is computed as `opex_specific_fixed[location, year] * capacity`,
    matching the formula enforced by `Technology.constraint_cost_opex_yearly`
    (technology.py:1875). Variable OPEX depends on dispatch and is not included.

    :param optimization_setup: live OptimizationSetup
    :param technology_name: name of the technology to evaluate
    :param location: node (or edge for transport technologies)
    :param year: yearly time step in `set_time_steps_yearly`
    :param capacity: capacity to be added
    :param storage_energy: True to evaluate the energy-side cost of a storage technology
    :return: (annualized_capex, annual_fixed_opex) as floats
    """
    tech = optimization_setup.get_element(Technology, technology_name)
    if tech is None:
        raise ValueError(f"Technology '{technology_name}' not found.")

    # CAPEX
    if isinstance(tech, StorageTechnology):
        capex = tech.calculate_capex_of_single_capacity(
            capacity, (location, year), storage_energy=storage_energy
        )
    else:
        capex = tech.calculate_capex_of_single_capacity(capacity, (location, year))

    # Fixed OPEX
    if isinstance(tech, StorageTechnology) and storage_energy:
        opex_specific_fixed = tech.opex_specific_fixed_energy
    else:
        opex_specific_fixed = tech.opex_specific_fixed
    opex_unit = opex_specific_fixed.loc[(location, year)]
    if hasattr(opex_unit, "iloc"):
        opex_unit = opex_unit.iloc[0]
    opex_fixed = float(opex_unit) * capacity

    return float(capex), float(opex_fixed)


def extract_duals(optimization_setup):
    """
    Extracts the shadow prices (dual variables) of the nodal energy balance constraint
    and returns the average dual variable per carrier and node.

    The average is the duration-weighted mean of the per-hour shadow prices over the
    optimized period: sum(dual_c) / sum(duration_c). The raw dual of an aggregated
    time step already equals duration * per-hour price, so summing the raw duals and
    dividing by total duration directly yields the per-hour average.

    :param optimization_setup: live OptimizationSetup with a solved Linopy model
    :return: pd.Series indexed by (carrier, node) with the average dual variable,
             or None if the constraint or duals are unavailable.
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

        # 1. Convert xarray to dataframe and unstack over operation time steps
        df_duals = duals.to_dataframe(name="dual").unstack("set_time_steps_operation")
        df_duals.columns = df_duals.columns.get_level_values("set_time_steps_operation")

        # 2. Duration-weighted average per (carrier, node)
        durations = pd.Series(
            optimization_setup.energy_system.time_steps.time_steps_operation_duration
        ).reindex(df_duals.columns)
        average_dual = df_duals.sum(axis=1) / durations.sum()
        average_dual.name = "average_dual"

        return average_dual
    except Exception as e:
        logging.error(f"Failed to extract shadow prices for {constraint_name}: {e}")
        return None

def investment_optimization(optimization_setup, step):
    # Dump the solved setup once so investment_optimization helpers can be tested
    # offline (see test_investment_optimization.py). Set ZEN_SKIP_DUMP=1 to skip.
    if not os.environ.get("ZEN_SKIP_DUMP") and not DEFAULT_OPTIMIZATION_SETUP_PICKLE.exists():
        try:
            save_optimization_setup(optimization_setup)
        except Exception as e:
            logging.warning(f"Could not pickle optimization_setup: {e}")

    extracted_duals = extract_duals(optimization_setup)
    logging.info(f"\n--- Market Prices (Average Duals) for Step {step} ---\n{extracted_duals}\n")

    # Example call for CAPEX and OPEX computation
    try:
        # Hole beispielhaft die erste gelistete Technologie und den ersten Node
        example_tech = list(optimization_setup.system.set_technologies)[0]
        example_node = list(optimization_setup.energy_system.set_nodes)[0]
        # Tatsächliches Kalenderjahr für diese Optimierungsschleife ermitteln (z.B. 2022 anstatt Index 0)
        example_idx = optimization_setup.energy_system.set_time_steps_yearly[0]
        example_year = optimization_setup.energy_system.set_time_steps_years[example_idx]

        capex, opex = compute_capex_and_opex_of_capacity(
            optimization_setup,
            technology_name=example_tech,
            location=example_node,
            year=example_year,
            capacity=1.0
        )
        logging.info(
            f"\n--- Investitions-Prüfung (Beispiel) ---\n"
            f"Technologie: {example_tech} | Ort: {example_node} | Jahr: {example_year} | Menge: 1.0\n"
            f"Annualisierte CAPEX: {capex:.2f}, Jährliche Fix-OPEX: {opex:.2f}\n"
            f"---------------------------------------\n"
        )
    except Exception as e:
        logging.warning(f"Konnte Beispiel-Kosten nicht ausgeben: {e}")


