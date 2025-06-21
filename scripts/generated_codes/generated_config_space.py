from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Define the configuration space for hyperparameter optimization.
    This configuration space is designed for a tabular dataset with 120 samples and 4 features.
    It includes hyperparameters for a simple classification model, such as k-NN or SVM.
    """
    cs = ConfigurationSpace()

    # --- k-Nearest Neighbors (k-NN) ---
    n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=1, upper=10, default_value=5)
    cs.add_hyperparameter(n_neighbors)

    # --- Support Vector Machine (SVM) ---
    C = UniformFloatHyperparameter("C", lower=0.1, upper=10.0, default_value=1.0, log=True)
    kernel = CategoricalHyperparameter("kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
    degree = UniformIntegerHyperparameter("degree", lower=2, upper=5, default_value=3)
    gamma = UniformFloatHyperparameter("gamma", lower=1e-4, upper=1.0, default_value=0.1, log=True)

    cs.add_hyperparameter(C)
    cs.add_hyperparameter(kernel)
    cs.add_hyperparameter(degree)
    cs.add_hyperparameter(gamma)

    return cs
