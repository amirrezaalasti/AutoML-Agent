from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    scenario = Scenario(
        configspace=cs,
        name="HyperparameterOptimization",
        output_directory="./logs/llama-3_3-70b-instruct_Iris_20250630_190234",
        deterministic=False,
        objectives="cost",
        crash_cost=1.0,
        termination_cost_threshold=1.0,
        n_trials=10,
        use_default_config=False,
        min_budget=1,
        max_budget=10,
        seed=0,
        n_workers=1,
    )
    return scenario
