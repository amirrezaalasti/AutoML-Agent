import os
from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs):
    output_dir = "./automl_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenario = Scenario(
        configspace=cs, name="smac_experiment", output_directory=output_dir, deterministic=False, n_workers=2, min_budget=1, max_budget=3
    )
    return scenario
