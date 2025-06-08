import os
from smac.scenario import Scenario
from ConfigSpace import ConfigurationSpace


def generate_scenario(cs: ConfigurationSpace) -> Scenario:
    """
    Generates a SMAC scenario for hyperparameter optimization, specifically tailored for image datasets.

    Args:
        cs (ConfigurationSpace): The configuration space to be optimized.

    Returns:
        Scenario: A configured SMAC Scenario object.
    """

    # Define the output directory for SMAC results
    output_dir = "./smac_output"
    os.makedirs(output_dir, exist_ok=True)

    # Determine the number of available cores.  Using os.cpu_count() may not be appropriate
    # in all environments (e.g., Docker containers).  Consider using a resource-aware method.
    n_workers = os.cpu_count() or 1  # Default to 1 worker if os.cpu_count() is None

    # Define budget settings: min_budget, max_budget
    # These depend heavily on the complexity of the model and the dataset size.
    # For an image dataset with 10 classes and ~60k examples,
    # these values need careful consideration.
    # Example: budget is based on the number of epochs/iterations
    min_budget = 3  # Minimum number of epochs/iterations
    max_budget = 27  # Maximum number of epochs/iterations

    # Set walltime and CPU time limits (in seconds)
    walltime_limit = 3600  # 1 hour

    # Define resource limits for each trial
    memory_limit = 4096  # 4GB memory limit (in MB)

    n_trials = 10  # Define n_trials, the number of trials

    scenario = Scenario(
        configspace=cs,
        name="image_dataset_optimization",
        output_directory=output_dir,
        deterministic=False,  # Enable for reproducibility during testing
        # but disable for better generalization during actual optimization
        n_trials=n_trials,
        walltime_limit=walltime_limit,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=n_workers,
    )

    # scenario.memory_limit = memory_limit # Setting memory limit after initialization - invalid

    return scenario
