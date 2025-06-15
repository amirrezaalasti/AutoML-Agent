from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace.conditions import InCondition, AndConjunction
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Returns a ConfigurationSpace object for hyperparameter optimization of a CNN
    for image classification.  The dataset is assumed to be a grayscale image
    dataset with 10 classes and images of size 28x28, similar to Fashion MNIST.
    """

    cs = ConfigurationSpace()

    # Model type: CNN or MLP
    model_type = CategoricalHyperparameter("model_type", choices=["cnn", "mlp"], default_value="cnn")
    cs.add_hyperparameter(model_type)

    # Batch size
    batch_size = CategoricalHyperparameter("batch_size", choices=[32, 64, 128, 256], default_value=128)
    cs.add_hyperparameter(batch_size)

    # Optimizer
    optimizer = CategoricalHyperparameter("optimizer", choices=["adam", "sgd"], default_value="adam")
    cs.add_hyperparameter(optimizer)

    # Learning rate
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
    cs.add_hyperparameter(learning_rate)

    # SGD-specific hyperparameters
    sgd_momentum = UniformFloatHyperparameter("sgd_momentum", lower=0.0, upper=0.9, default_value=0.0)
    cs.add_hyperparameter(sgd_momentum)

    # Condition for SGD momentum
    momentum_condition = InCondition(sgd_momentum, optimizer, ["sgd"])
    cs.add_condition(momentum_condition)

    # CNN specific hyperparameters
    num_conv_layers = UniformIntegerHyperparameter("num_conv_layers", lower=1, upper=3, default_value=2)
    cs.add_hyperparameter(num_conv_layers)

    # Number of filters for conv layers
    conv_filters_1 = CategoricalHyperparameter("conv_filters_1", choices=[32, 64, 128], default_value=64)
    cs.add_hyperparameter(conv_filters_1)

    conv_filters_2 = CategoricalHyperparameter("conv_filters_2", choices=[64, 128, 256], default_value=128)
    cs.add_hyperparameter(conv_filters_2)

    conv_filters_3 = CategoricalHyperparameter("conv_filters_3", choices=[128, 256, 512], default_value=256)
    cs.add_hyperparameter(conv_filters_3)

    # Kernel size for conv layers
    kernel_size = CategoricalHyperparameter("kernel_size", choices=[3, 5], default_value=3)
    cs.add_hyperparameter(kernel_size)

    # Pooling type
    pooling_type = CategoricalHyperparameter("pooling_type", choices=["max", "avg"], default_value="max")
    cs.add_hyperparameter(pooling_type)

    # Number of dense layers
    num_dense_layers = UniformIntegerHyperparameter("num_dense_layers", lower=1, upper=2, default_value=1)
    cs.add_hyperparameter(num_dense_layers)

    # Size of dense layers
    dense_size_1 = CategoricalHyperparameter("dense_size_1", choices=[128, 256, 512], default_value=256)
    cs.add_hyperparameter(dense_size_1)

    dense_size_2 = CategoricalHyperparameter("dense_size_2", choices=[256, 512, 1024], default_value=512)
    cs.add_hyperparameter(dense_size_2)

    # Dropout rate
    dropout_rate = UniformFloatHyperparameter("dropout_rate", lower=0.0, upper=0.5, default_value=0.2)
    cs.add_hyperparameter(dropout_rate)

    # Normalization Strategy
    normalization_strategy = CategoricalHyperparameter(
        "normalization_strategy", choices=["none", "batchnorm", "layernorm"], default_value="batchnorm"
    )
    cs.add_hyperparameter(normalization_strategy)

    # CNN Condition
    cnn_condition = InCondition(child=num_conv_layers, parent=model_type, values=["cnn"])
    cs.add_condition(cnn_condition)

    cnn_condition_filters1 = InCondition(child=conv_filters_1, parent=model_type, values=["cnn"])
    cs.add_condition(cnn_condition_filters1)

    cnn_condition_kernel = InCondition(child=kernel_size, parent=model_type, values=["cnn"])
    cs.add_condition(cnn_condition_kernel)

    cnn_condition_pool = InCondition(child=pooling_type, parent=model_type, values=["cnn"])
    cs.add_condition(cnn_condition_pool)

    dense_condition_1 = InCondition(child=dense_size_1, parent=model_type, values=["mlp"])
    cs.add_condition(dense_condition_1)

    dense_condition_2 = InCondition(child=dense_size_2, parent=model_type, values=["mlp"])
    cs.add_condition(dense_condition_2)

    dense_condition_layers = InCondition(child=num_dense_layers, parent=model_type, values=["mlp"])
    cs.add_condition(dense_condition_layers)

    drop_condition = InCondition(child=dropout_rate, parent=model_type, values=["mlp"])
    cs.add_condition(drop_condition)

    norm_condition = InCondition(child=normalization_strategy, parent=model_type, values=["mlp"])
    cs.add_condition(norm_condition)

    # Conditions for number of conv layers
    conv_filters_2_condition = AndConjunction(
        InCondition(child=conv_filters_2, parent=num_conv_layers, values=[2, 3]), InCondition(child=conv_filters_2, parent=model_type, values=["cnn"])
    )
    cs.add_condition(conv_filters_2_condition)

    conv_filters_3_condition = AndConjunction(
        InCondition(child=conv_filters_3, parent=num_conv_layers, values=[3]), InCondition(child=conv_filters_3, parent=model_type, values=["cnn"])
    )
    cs.add_condition(conv_filters_3_condition)

    return cs
