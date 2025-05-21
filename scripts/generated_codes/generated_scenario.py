from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs):
    scenario = Scenario(configspace=cs, output_directory="./automl_results", deterministic=False, n_workers=4, min_budget=1, max_budget=10)
    return scenario
