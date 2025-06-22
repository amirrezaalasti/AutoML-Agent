from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, InCondition
from typing import Any


def get_configspace() -> ConfigurationSpace:
    """
    Defines the configuration space for a CNN model for image classification.
    """
    cs = ConfigurationSpace()

    # Optimizer
    optimizer = CategoricalHyperparameter("optimizer", ["adam", "sgd", "rmsprop"], default_value="adam")
    cs.add_hyperparameter(optimizer)

    # Learning Rate
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
    cs.add_hyperparameter(learning_rate)

    # Batch Size
    batch_size = UniformIntegerHyperparameter("batch_size", lower=32, upper=256, default_value=64, log=True)
    cs.add_hyperparameter(batch_size)

    # Epochs
    epochs = UniformIntegerHyperparameter("epochs", lower=10, upper=100, default_value=20, log=False)
    cs.add_hyperparameter(epochs)

    # Momentum (only relevant for SGD)
    momentum = UniformFloatHyperparameter("momentum", lower=0.0, upper=0.99, default_value=0.9)
    cs.add_hyperparameter(momentum)

    # Dropout Rate
    dropout_rate = UniformFloatHyperparameter("dropout_rate", lower=0.0, upper=0.5, default_value=0.2)
    cs.add_hyperparameter(dropout_rate)

    # Number of Convolutional Layers
    num_conv_layers = UniformIntegerHyperparameter("num_conv_layers", lower=1, upper=3, default_value=2, log=False)
    cs.add_hyperparameter(num_conv_layers)

    # Number of Filters for Conv Layers
    num_filters_l1 = UniformIntegerHyperparameter("num_filters_l1", lower=16, upper=128, default_value=32, log=True)
    num_filters_l2 = UniformIntegerHyperparameter("num_filters_l2", lower=16, upper=128, default_value=64, log=True)
    num_filters_l3 = UniformIntegerHyperparameter("num_filters_l3", lower=16, upper=128, default_value=64, log=True)
    cs.add_hyperparameter(num_filters_l1)
    cs.add_hyperparameter(num_filters_l2)
    cs.add_hyperparameter(num_filters_l3)

    # Kernel Size for Conv Layers
    kernel_size = CategoricalHyperparameter("kernel_size", choices=[3, 5], default_value=3)
    cs.add_hyperparameter(kernel_size)

    # Add conditions
    momentum_condition = InCondition(child=momentum, parent=optimizer, values=["sgd"])
    cs.add_condition(momentum_condition)

    return cs
