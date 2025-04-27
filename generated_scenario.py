from smac.scenario import Scenario

def generate_scenario(cs):
    scenario = Scenario({
        'run_obj': 'quality',
        'runcount-limit': 100,
        'cs': cs,
        'output_dir': "./automl_results",
        'shared_model': False,
        'limit_resources': False,
        'wallclock_limit': 3600,
        'cutoff': 60
    })
    return scenario
