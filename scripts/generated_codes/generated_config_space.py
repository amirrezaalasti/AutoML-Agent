from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause


def get_configspace():
    cs = ConfigurationSpace()

    # Define hyperparameters
    n_estimators = Integer("n_estimators", bounds=(10, 200), default=100)
    max_depth = Integer("max_depth", bounds=(2, 10), default=5)
    min_samples_split = Integer("min_samples_split", bounds=(2, 20), default=2)
    min_samples_leaf = Integer("min_samples_leaf", bounds=(1, 10), default=1)

    # Add hyperparameters to the configuration space
    cs.add([n_estimators, max_depth, min_samples_split, min_samples_leaf])

    # Define forbidden clauses
    forbidden_clause = ForbiddenAndConjunction(ForbiddenEqualsClause(max_depth, 2), ForbiddenEqualsClause(min_samples_split, 20))

    # Add forbidden clauses to the configuration space
    cs.add_forbidden_clause(forbidden_clause)

    return cs
