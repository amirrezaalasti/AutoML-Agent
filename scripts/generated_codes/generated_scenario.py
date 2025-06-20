from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    scenario = Scenario(
        configspace=cs,
        name="HyperparameterOptimization",
        output_directory="./logs/gemini-2.0-flash_credit-g_20250620_112642",
        deterministic=False,
        n_trials=10,
        n_workers=1,
        min_budget=1,
        max_budget=3,
    )
    return scenario
