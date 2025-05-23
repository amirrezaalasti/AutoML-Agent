from smac.scenario import Scenario


def generate_scenario(cs):
    scenario = Scenario(
        configspace=cs,
        output_directory="./automl_results",
        deterministic=False,
        n_workers=4,
        min_budget=1,
        max_budget=100,
    )
    return scenario
