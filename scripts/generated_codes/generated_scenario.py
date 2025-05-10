from smac import Scenario

def generate_scenario(cs):
    scenario = Scenario(
        configspace=cs,
        objectives=["validation_loss"],
        output_directory="./automl_results",
        deterministic=False,
        n_workers=4,
        min_budget=1,
        max_budget=100,
    )
    return scenario
