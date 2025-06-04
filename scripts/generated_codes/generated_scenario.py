import os
from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs):
    """
    Generates a SMAC scenario object.
    """
    scenario = Scenario(configspace=cs, output_directory="./automl_results", deterministic=False, n_workers=2, min_budget=1, max_budget=3)
    return scenario
