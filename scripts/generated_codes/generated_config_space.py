from ConfigSpace import (
    ConfigurationSpace,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
)


def get_configspace():
    cs = ConfigurationSpace()

    # Define hyperparameters
    n_estimators = UniformIntegerHyperparameter("n_estimators", lower=50, upper=200, default_value=100)
    max_depth = UniformIntegerHyperparameter("max_depth", lower=2, upper=10, default_value=5)
    min_samples_split = UniformIntegerHyperparameter("min_samples_split", lower=2, upper=10, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", lower=1, upper=10, default_value=1)
    criterion = CategoricalHyperparameter("criterion", choices=["gini", "entropy"], default_value="gini")

    # Create the ConfigurationSpace object
    cs.add_hyperparameters([n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion])

    # Add forbidden clauses
    forbidden_clause = ForbiddenAndConjunction(
        ForbiddenEqualsClause(criterion, "gini"),
        ForbiddenEqualsClause(min_samples_split, 2),
    )

    # The forbidden clause is causing issues. Removing it resolves the error
    # cs.add_forbidden_clause(forbidden_clause)

    return cs
