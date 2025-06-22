from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    scenario = Scenario(
        configspace=cs,
        name="HyperparameterOptimization",
        output_directory="./logs/gemini-2.0-flash_MNIST_20250622_115013",
        deterministic=False,
        n_trials=10,
        n_workers=1,
        use_default_config=True,
    )
    return scenario
