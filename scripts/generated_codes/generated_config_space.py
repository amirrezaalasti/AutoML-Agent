from ConfigSpace import (
    ConfigurationSpace,
    Categorical,
    Float,
    Integer,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
)


def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    # Define hyperparameters
    optimizer = Categorical("optimizer", ["sgd", "adam"], default="adam")
    learning_rate = Categorical("learning_rate", ["constant", "adaptive"], default="constant")
    eta0 = Float("eta0", (1e-5, 1e-1), default=1e-3, log=True)
    batch_size = Integer("batch_size", (32, 256), default=64, log=True)
    num_layers = Integer("num_layers", (1, 3), default=2)
    num_units = Integer("num_units", (64, 512), default=128, log=True)

    # Add hyperparameters to the configuration space
    cs.add([optimizer, learning_rate, eta0, batch_size, num_layers, num_units])

    # Add conditions
    condition_eta0 = EqualsCondition(eta0, learning_rate, "constant")
    cs.add_condition(condition_eta0)

    # Add forbidden clauses
    forbidden_clause = ForbiddenAndConjunction(
        ForbiddenEqualsClause(optimizer, "sgd"),
        ForbiddenEqualsClause(learning_rate, "adaptive"),
    )
    cs.add_forbidden_clause(forbidden_clause)

    return cs
