from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


def get_configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace()

    # Learning rate
    cs.add_hyperparameter(UniformFloatHyperparameter(name="learning_rate", lower=0.001, upper=0.1, default_value=0.01, log=True))

    # Regularization strength
    cs.add_hyperparameter(UniformFloatHyperparameter(name="regularization_strength", lower=0.01, upper=1.0, default_value=0.1, log=True))

    # Number of hidden layers
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="num_hidden_layers", lower=1, upper=5, default_value=2))

    # Number of units in each hidden layer
    cs.add_hyperparameter(UniformIntegerHyperparameter(name="num_units_in_hidden_layer", lower=10, upper=100, default_value=50))

    # Dropout rate
    cs.add_hyperparameter(UniformFloatHyperparameter(name="dropout_rate", lower=0.1, upper=0.5, default_value=0.2))

    # Activation function
    cs.add_hyperparameter(CategoricalHyperparameter(name="activation_function", choices=["relu", "tanh", "sigmoid"], default_value="relu"))

    # Optimizer
    cs.add_hyperparameter(CategoricalHyperparameter(name="optimizer", choices=["adam", "sgd", "rmsprop"], default_value="adam"))

    return cs
