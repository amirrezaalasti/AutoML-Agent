from smac.scenario import Scenario
from ConfigSpace import ConfigurationSpace

def generate_scenario(cs):
    scenario = Scenario({
        'run_obj': 'quality',
        'runcount-limit': 100,
        'cs': cs,
        'output_dir': "./automl_results",
        'shared_model': False
    })
    return scenario
