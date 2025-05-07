from smac import Scenario

def generate_scenario(cs):
    scenario = Scenario({
        "run_obj": "quality",
        "output_dir": "./automl_results",
        "shared_model": False,
        "multi_objectives": ["validation_loss"],
        "abort_on_first_run_crash": True,
        "wallclock_limit": 3600.0,
        "ta_run_limit": 100,
        "memory_limit": 16000.0,
        "cutoff": 3600.0,
        "deterministic": False
    })
    return scenario
