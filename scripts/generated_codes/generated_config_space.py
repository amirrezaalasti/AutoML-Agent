from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    InCondition,
    AndConjunction,
)


def get_configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace()

    # Define hyperparameters for SVM
    kernel = CategoricalHyperparameter("svm_kernel", ["linear", "rbf", "poly"], default_value="rbf")
    cs.add_hyperparameter(kernel)

    C = UniformFloatHyperparameter("svm_C", lower=0.01, upper=10, default_value=1.0, log=True)
    cs.add_hyperparameter(C)

    gamma = UniformFloatHyperparameter("svm_gamma", lower=0.001, upper=1.0, default_value=0.1, log=True)
    cs.add_hyperparameter(gamma)

    degree = UniformIntegerHyperparameter("svm_degree", lower=2, upper=5, default_value=3)
    cs.add_hyperparameter(degree)

    # Add conditions
    condition_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly"])
    cs.add_condition(condition_gamma)

    condition_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    cs.add_condition(condition_degree)

    return cs
