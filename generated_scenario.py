from smac import Scenario

def generate_scenario(cs):
    scenario = Scenario({
        "run_obj": "quality",
        "output_dir": "./automl_results",
        "shared_model": False,
        "multi_objectives": ["validation_loss"],
        "overall_obj": "validation_loss",
        "deterministic": False,
        "wallclock_limit": 3600.0,
        "abort_on_first_run_crash": True,
        "limit_resources": True,
        "memory_limit": 16000.0,
        "cutoff": 300.0,
        "ta_run_limit": 100,
        "intens_min_chall": 1,
        "intens_adaptive_capping_slackfactor": 1.5,
        "rf_num_trees": 10,
        "rf_max_depth": 5,
        "rf_min_samples_leaf": 1,
        "rf_min_samples_split": 2,
        "rf_ratio_features": 0.5,
        "save_results_instantly": True,
    })
    return scenario
