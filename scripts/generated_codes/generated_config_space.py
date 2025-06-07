from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause


def get_configspace():
    cs = ConfigurationSpace()

    # Define hyperparameters for a RandomForestClassifier
    n_estimators = Integer("n_estimators", bounds=(10, 200), default=100)
    max_depth = Integer("max_depth", bounds=(2, 10), default=6)
    min_samples_split = Integer("min_samples_split", bounds=(2, 10), default=2)
    min_samples_leaf = Integer("min_samples_leaf", bounds=(1, 10), default=1)

    cs.add_hyperparameters([n_estimators, max_depth, min_samples_split, min_samples_leaf])

    # Example of a forbidden clause
    forbidden_clause = ForbiddenAndConjunction(ForbiddenEqualsClause(max_depth, 2), ForbiddenEqualsClause(min_samples_split, 2))

    cs.add_forbidden_clause(forbidden_clause)

    return cs
