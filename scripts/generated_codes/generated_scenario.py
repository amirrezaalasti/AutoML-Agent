from smac import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    scenario = Scenario(
        configspace=cs,
        name="wave_energy_regression",
        output_directory="./logs/gemini-2.0-flash_wave_energy_20250627_001956",
        deterministic=False,
        n_trials=10,
        n_workers=1,
    )
    return scenario
