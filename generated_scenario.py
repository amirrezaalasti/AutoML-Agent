from smac import Scenario

def generate_scenario(cs):
    scenario = Scenario({
        "run_obj": "quality",
        "output_dir": "./automl_results",
        "shared_model": False,
        "deterministic": False,
        "limit_resources": True,
        "cs": cs
    })
    return scenario
