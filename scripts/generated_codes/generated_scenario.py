import logging
from pathlib import Path

from smac.scenario import Scenario


def generate_scenario(cs):
    return Scenario(
        configspace=cs,
        name="gemini-2.0-flashiris20250608_105539",
        output_directory="./automl_results",
        deterministic=False,
        n_workers=2,
        min_budget=1,
        max_budget=3,
        n_trials=10,
    )
