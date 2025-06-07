from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs):
    """
    Generates a Scenario object for SMAC.

    Args:
        cs (ConfigurationSpace): The configuration space to be used.

    Returns:
        Scenario: A configured Scenario object.
    """
    scenario = Scenario(
        configspace=cs,
        name="gemini-2.0-flashsunspots_(statsmodels)20250607_140029",
        output_directory="./automl_results",
        deterministic=False,
        n_workers=2,
        min_budget=1,
        max_budget=9,
        n_trials=10,
    )
    return scenario
