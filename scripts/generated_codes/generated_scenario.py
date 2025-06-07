import ConfigSpace as CS
from smac.scenario import Scenario


def generate_scenario(cs):
    scenario = Scenario(
        configspace=cs,
        name="gemini-2.0-flashbreast_cancer20250607_101039",
        output_directory="./automl_results",
        deterministic=False,
        n_workers=4,
        min_budget=1,
        max_budget=9,
        n_trials=10,
    )
    return scenario
