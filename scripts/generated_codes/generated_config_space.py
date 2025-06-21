from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Returns a ConfigurationSpace object for hyperparameter optimization.
    This configuration space is tailored for a tabular dataset with mixed
    categorical and numerical features, suitable for classification.
    """
    cs = ConfigurationSpace()

    # Define hyperparameters for a Gradient Boosting Machine (GBM)
    # Adjust ranges based on dataset characteristics

    # Learning Rate
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.001, upper=0.3, default_value=0.1, log=True)
    cs.add_hyperparameter(learning_rate)

    # Number of Estimators (Boosting Rounds)
    n_estimators = UniformIntegerHyperparameter("n_estimators", lower=50, upper=500, default_value=100)
    cs.add_hyperparameter(n_estimators)

    # Maximum Depth of Trees
    max_depth = UniformIntegerHyperparameter("max_depth", lower=2, upper=10, default_value=3)
    cs.add_hyperparameter(max_depth)

    # Minimum Samples Split
    min_samples_split = UniformIntegerHyperparameter("min_samples_split", lower=2, upper=20, default_value=2)
    cs.add_hyperparameter(min_samples_split)

    # Minimum Samples Leaf
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", lower=1, upper=10, default_value=1)
    cs.add_hyperparameter(min_samples_leaf)

    # Subsample Ratio
    subsample = UniformFloatHyperparameter("subsample", lower=0.5, upper=1.0, default_value=1.0)
    cs.add_hyperparameter(subsample)

    # L2 Regularization (L2 penalty)
    l2_regularization = UniformFloatHyperparameter("l2_regularization", lower=1e-9, upper=1.0, default_value=1e-9, log=True)
    cs.add_hyperparameter(l2_regularization)

    # Define hyperparameters for preprocessing (OneHotEncoder)
    use_one_hot_encoding = CategoricalHyperparameter("use_one_hot_encoding", choices=[True, False], default_value=True)
    cs.add_hyperparameter(use_one_hot_encoding)

    return cs
