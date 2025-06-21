from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    scenario = Scenario(
        configspace=cs,
        name="HyperparameterOptimization",
        output_directory="./logs/gemini-2.0-flash_Iris_20250621_103700",
        deterministic=False,
        n_trials=10,
        n_workers=1,
        crash_cost=100000000.0,
    )
    return scenario
