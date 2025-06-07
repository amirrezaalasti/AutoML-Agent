from smac.scenario import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs):
    scenario = Scenario(
        configspace=cs,
        name="gemini-2.0-flashiris20250607_112729",
        output_directory="./automl_results",
        deterministic=False,
        n_workers=4,
        min_budget=1,
        max_budget=10,
        n_trials=10,
    )
    return scenario
