from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition, OrConjunction


def get_configspace() -> ConfigurationSpace:
    """
    Defines the configuration space for hyperparameter optimization.
    """
    cs = ConfigurationSpace()

    # Optimizer
    optimizer = CategoricalHyperparameter("optimizer", choices=["Adam", "SGD", "RMSprop"], default_value="Adam")
    cs.add_hyperparameter(optimizer)

    # Learning Rate (Conditional on Optimizer)
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
    cs.add_hyperparameter(learning_rate)
    optimizer_cond_adam = InCondition(learning_rate, optimizer, ["Adam"])
    optimizer_cond_sgd = InCondition(learning_rate, optimizer, ["SGD"])
    optimizer_cond_rmsprop = InCondition(learning_rate, optimizer, ["RMSprop"])
    optimizer_cond = OrConjunction(optimizer_cond_adam, optimizer_cond_sgd, optimizer_cond_rmsprop)
    cs.add_condition(optimizer_cond)

    # Batch Size
    batch_size = UniformIntegerHyperparameter("batch_size", lower=8, upper=64, default_value=32, log=True)
    cs.add_hyperparameter(batch_size)

    # Number of Layers (NN Specific)
    num_layers = UniformIntegerHyperparameter("num_layers", lower=1, upper=3, default_value=2)
    cs.add_hyperparameter(num_layers)

    # Dropout Rate (NN Specific)
    dropout_rate = UniformFloatHyperparameter("dropout_rate", lower=0.0, upper=0.5, default_value=0.2)
    cs.add_hyperparameter(dropout_rate)

    # Activation Function (NN Specific)
    activation = CategoricalHyperparameter("activation", choices=["ReLU", "Tanh", "Sigmoid"], default_value="ReLU")
    cs.add_hyperparameter(activation)

    # Number of Estimators (GB Specific)
    num_estimators = UniformIntegerHyperparameter("num_estimators", lower=50, upper=200, default_value=100)
    cs.add_hyperparameter(num_estimators)

    # Max Depth (GB Specific)
    max_depth = UniformIntegerHyperparameter("max_depth", lower=2, upper=6, default_value=3)
    cs.add_hyperparameter(max_depth)

    # Kernel (SVM specific)
    kernel = CategoricalHyperparameter("kernel", choices=["linear", "rbf", "poly"], default_value="rbf")
    cs.add_hyperparameter(kernel)

    # C (SVM and Logistic Regression)
    C = UniformFloatHyperparameter("C", lower=0.1, upper=10.0, default_value=1.0, log=True)
    cs.add_hyperparameter(C)

    return cs
