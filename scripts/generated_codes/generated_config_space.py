from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, InCondition
import numpy as np


def get_configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace()

    # Model-specific hyperparameters
    cs.add_hyperparameter(UniformFloatHyperparameter("learning_rate", 0.001, 0.1, default_value=0.01))
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_layers", 1, 5, default_value=2))
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_units", 32, 512, default_value=128))

    # Regularization parameters
    cs.add_hyperparameter(UniformFloatHyperparameter("dropout_rate", 0.0, 0.5, default_value=0.2))
    cs.add_hyperparameter(UniformFloatHyperparameter("l1_regularization", 0.0, 0.1, default_value=0.01))
    cs.add_hyperparameter(UniformFloatHyperparameter("l2_regularization", 0.0, 0.1, default_value=0.01))

    # Architecture parameters
    cs.add_hyperparameter(CategoricalHyperparameter("activation", ["relu", "tanh", "sigmoid"], default_value="relu"))
    cs.add_hyperparameter(CategoricalHyperparameter("optimizer", ["adam", "sgd", "rmsprop"], default_value="adam"))

    # Optimization parameters
    cs.add_hyperparameter(UniformIntegerHyperparameter("batch_size", 32, 1024, default_value=128))
    cs.add_hyperparameter(UniformIntegerHyperparameter("epochs", 10, 100, default_value=50))

    # Conditional hyperparameters
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_units_layer_1", 32, 512, default_value=128))
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_units_layer_2", 32, 512, default_value=128))

    # Add conditions for conditional hyperparameters
    n_layers = cs.get_hyperparameter("n_layers")
    n_units_layer_1 = cs.get_hyperparameter("n_units_layer_1")
    n_units_layer_2 = cs.get_hyperparameter("n_units_layer_2")

    cs.add_condition(InCondition(child=n_units_layer_1, parent=n_layers, values=[2, 3, 4, 5]))
    cs.add_condition(InCondition(child=n_units_layer_2, parent=n_layers, values=[3, 4, 5]))

    return cs
