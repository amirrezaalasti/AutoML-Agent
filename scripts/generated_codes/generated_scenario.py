from smac import Scenario
from ConfigSpace import ConfigurationSpace
from pathlib import Path


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    """
    Generates a SMAC scenario configuration for hyperparameter optimization.

    Args:
        cs (ConfigurationSpace): The configuration space from which to sample configurations.

    Returns:
        Scenario: A configured SMAC Scenario object.
    """

    scenario = Scenario(
        configspace=cs,
        name="image_classification_experiment",
        output_directory=Path("automl_results"),
        deterministic=False,
        n_trials=10,
        n_workers=1,
        min_budget=1,
        max_budget=5,
    )
    return scenario
