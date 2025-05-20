from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace import EqualsCondition, InCondition, OrConjunction


def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    # Define hyperparameters
    optimizer = Categorical("optimizer", ["Adam", "SGD"], default="Adam")
    learning_rate = Float("learning_rate", (1e-5, 1e-2), log=True, default=1e-3)
    batch_size = Categorical("batch_size", [32, 64, 128], default=32)
    num_layers = Integer("num_layers", (1, 5), default=2)
    num_units = Integer("num_units", (32, 256), log=True, default=64)
    dropout = Float("dropout", (0.0, 0.9), default=0.5)

    # Add hyperparameters to the configuration space
    cs.add_hyperparameters([optimizer, learning_rate, batch_size, num_layers, num_units, dropout])

    # Define forbidden clauses
    # Example: If optimizer is SGD, then learning rate must be less than 0.001
    forbidden_clause = ForbiddenAndConjunction(ForbiddenEqualsClause(optimizer, "SGD"), ForbiddenEqualsClause(learning_rate, 0.01))
    cs.add_forbidden_clause(forbidden_clause)

    return cs
