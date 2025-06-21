from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Define the configuration space for machine learning hyperparameter optimization.
    This configuration space includes hyperparameters for KNN and SVM classifiers,
    suitable for the given tabular dataset with 120 samples and 4 features.
    """
    cs = ConfigurationSpace()

    # KNN Hyperparameters
    n_neighbors = UniformIntegerHyperparameter("knn__n_neighbors", lower=1, upper=20, default_value=5)
    cs.add_hyperparameter(n_neighbors)

    weights = CategoricalHyperparameter("knn__weights", choices=["uniform", "distance"], default_value="uniform")
    cs.add_hyperparameter(weights)

    # SVM Hyperparameters
    svm_kernel = CategoricalHyperparameter("svm__kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
    cs.add_hyperparameter(svm_kernel)

    svm_C = UniformFloatHyperparameter("svm__C", lower=0.001, upper=1000.0, default_value=1.0, log=True)
    cs.add_hyperparameter(svm_C)

    svm_gamma = UniformFloatHyperparameter("svm__gamma", lower=0.0001, upper=10.0, default_value=0.1, log=True)
    cs.add_hyperparameter(svm_gamma)

    svm_degree = UniformIntegerHyperparameter("svm__degree", lower=2, upper=5, default_value=3)
    cs.add_hyperparameter(svm_degree)

    return cs
