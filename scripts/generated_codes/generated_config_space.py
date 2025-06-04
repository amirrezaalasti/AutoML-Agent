from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause


def get_configspace():
    cs = ConfigurationSpace()

    # Define hyperparameters
    optimizer = Categorical("optimizer", ["Adam", "SGD", "RMSprop"], default="Adam")
    learning_rate = Float("learning_rate", (1e-5, 1e-2), log=True, default=1e-3)
    batch_size = Categorical("batch_size", [32, 64, 128], default=32)
    num_epochs = Integer("num_epochs", (10, 50), default=20)

    # CNN specific parameters
    use_cnn = Categorical("use_cnn", [True, False], default=True)
    num_filters = Integer("num_filters", (32, 128), default=64, log=True)
    kernel_size = Categorical("kernel_size", [3, 5], default=3)

    # Dense layer specific parameters
    num_units = Integer("num_units", (64, 512), default=256, log=True)

    # Add hyperparameters to the ConfigurationSpace
    cs.add([optimizer, learning_rate, batch_size, num_epochs, use_cnn, num_filters, kernel_size, num_units])

    # Add forbidden clauses
    # Example: If not using CNN, num_filters and kernel_size are invalid
    forbidden_cnn_params = ForbiddenAndConjunction(
        ForbiddenEqualsClause(use_cnn, False), ForbiddenEqualsClause(num_filters, 64)  # Just a dummy value, the actual value doesn't matter
    )

    cs.add_forbidden_clause(forbidden_cnn_params)

    return cs
