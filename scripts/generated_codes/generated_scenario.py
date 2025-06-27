from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    scenario = Scenario(
        configspace=cs,
        name="HyperparameterOptimization",
        output_directory="./logs/llama-3.3-70b-instruct_wave_energy_20250627_115252",
        deterministic=False,
        objectives=["cost"],
        crash_cost=float("inf"),
        termination_cost_threshold=float("inf"),
        n_trials=10,
        use_default_config=False,
        min_budget=1,
        max_budget=100,
        seed=0,
        n_workers=1,
    )
    return scenario
