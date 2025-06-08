from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Creates a configuration space for a CNN model suitable for image classification.
    """
    cs = ConfigurationSpace()

    # Data Augmentation
    data_augmentation = CategoricalHyperparameter("data_augmentation", ["none", "horizontal_flip", "random_rotation"], default_value="none")
    cs.add_hyperparameter(data_augmentation)

    random_rotation_degree = UniformIntegerHyperparameter("random_rotation_degree", lower=1, upper=45, default_value=10)
    cs.add_hyperparameter(random_rotation_degree)

    condition_rotation = InCondition(random_rotation_degree, data_augmentation, ["random_rotation"])
    cs.add_condition(condition_rotation)

    # Batch Size
    batch_size = UniformIntegerHyperparameter("batch_size", lower=32, upper=256, default_value=128)
    cs.add_hyperparameter(batch_size)

    # Epochs
    epochs = UniformIntegerHyperparameter("epochs", lower=5, upper=50, default_value=10)
    cs.add_hyperparameter(epochs)

    # Learning Rate
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-4, upper=1e-2, default_value=1e-3, log=True)
    cs.add_hyperparameter(learning_rate)

    # Optimizer
    optimizer = CategoricalHyperparameter("optimizer", ["adam", "sgd", "rmsprop"], default_value="adam")
    cs.add_hyperparameter(optimizer)

    # Adam Specific Parameters
    adam_beta1 = UniformFloatHyperparameter("adam_beta1", lower=0.8, upper=0.999, default_value=0.9)
    cs.add_hyperparameter(adam_beta1)
    adam_beta2 = UniformFloatHyperparameter("adam_beta2", lower=0.9, upper=0.9999, default_value=0.999)
    cs.add_hyperparameter(adam_beta2)

    condition_adam_beta1 = InCondition(child=adam_beta1, parent=optimizer, values=["adam"])
    cs.add_condition(condition_adam_beta1)
    condition_adam_beta2 = InCondition(child=adam_beta2, parent=optimizer, values=["adam"])
    cs.add_condition(condition_adam_beta2)

    # SGD Specific Parameters
    sgd_momentum = UniformFloatHyperparameter("sgd_momentum", lower=0.0, upper=0.99, default_value=0.0)
    cs.add_hyperparameter(sgd_momentum)
    condition_sgd_momentum = InCondition(child=sgd_momentum, parent=optimizer, values=["sgd"])
    cs.add_condition(condition_sgd_momentum)

    # RMSprop Specific Parameters
    rmsprop_rho = UniformFloatHyperparameter("rmsprop_rho", lower=0.9, upper=0.999, default_value=0.9)
    cs.add_hyperparameter(rmsprop_rho)
    condition_rmsprop_rho = InCondition(child=rmsprop_rho, parent=optimizer, values=["rmsprop"])
    cs.add_condition(condition_rmsprop_rho)

    # Convolutional Layers
    num_conv_layers = UniformIntegerHyperparameter("num_conv_layers", lower=1, upper=5, default_value=3)
    cs.add_hyperparameter(num_conv_layers)

    # Filters for each convolutional layer
    filters_layer_1 = UniformIntegerHyperparameter("filters_layer_1", lower=32, upper=256, default_value=64, log=True)
    cs.add_hyperparameter(filters_layer_1)
    filters_layer_2 = UniformIntegerHyperparameter("filters_layer_2", lower=32, upper=256, default_value=64, log=True)
    cs.add_hyperparameter(filters_layer_2)
    filters_layer_3 = UniformIntegerHyperparameter("filters_layer_3", lower=32, upper=256, default_value=128, log=True)
    cs.add_hyperparameter(filters_layer_3)
    filters_layer_4 = UniformIntegerHyperparameter("filters_layer_4", lower=32, upper=256, default_value=128, log=True)
    cs.add_hyperparameter(filters_layer_4)
    filters_layer_5 = UniformIntegerHyperparameter("filters_layer_5", lower=32, upper=256, default_value=256, log=True)
    cs.add_hyperparameter(filters_layer_5)

    # Kernel Size
    kernel_size = UniformIntegerHyperparameter("kernel_size", lower=3, upper=5, default_value=3)
    cs.add_hyperparameter(kernel_size)

    # Pooling Type
    pooling_type = CategoricalHyperparameter("pooling_type", ["max", "average"], default_value="max")
    cs.add_hyperparameter(pooling_type)

    # Dense Layers
    num_dense_layers = UniformIntegerHyperparameter("num_dense_layers", lower=1, upper=3, default_value=2)
    cs.add_hyperparameter(num_dense_layers)

    dense_units_1 = UniformIntegerHyperparameter("dense_units_1", lower=64, upper=512, default_value=256, log=True)
    cs.add_hyperparameter(dense_units_1)

    dense_units_2 = UniformIntegerHyperparameter("dense_units_2", lower=64, upper=512, default_value=128, log=True)
    cs.add_hyperparameter(dense_units_2)

    dense_units_3 = UniformIntegerHyperparameter("dense_units_3", lower=64, upper=512, default_value=64, log=True)
    cs.add_hyperparameter(dense_units_3)

    # Dropout Rate
    dropout_rate = UniformFloatHyperparameter("dropout_rate", lower=0.0, upper=0.5, default_value=0.25)
    cs.add_hyperparameter(dropout_rate)

    # Conditions for number of layers
    condition_layer_2 = InCondition(child=filters_layer_2, parent=num_conv_layers, values=[2, 3, 4, 5])
    cs.add_condition(condition_layer_2)

    condition_layer_3 = InCondition(child=filters_layer_3, parent=num_conv_layers, values=[3, 4, 5])
    cs.add_condition(condition_layer_3)

    condition_layer_4 = InCondition(child=filters_layer_4, parent=num_conv_layers, values=[4, 5])
    cs.add_condition(condition_layer_4)

    condition_layer_5 = InCondition(child=filters_layer_5, parent=num_conv_layers, values=[5])
    cs.add_condition(condition_layer_5)

    condition_dense_layer_2 = InCondition(child=dense_units_2, parent=num_dense_layers, values=[2, 3])
    cs.add_condition(condition_dense_layer_2)

    condition_dense_layer_3 = InCondition(child=dense_units_3, parent=num_dense_layers, values=[3])
    cs.add_condition(condition_dense_layer_3)

    return cs
