from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause


def get_configspace():
    cs = ConfigurationSpace()

    # Define hyperparameters for a simple classifier (e.g., Support Vector Machine)
    kernel = Categorical("kernel", ["linear", "rbf", "poly", "sigmoid"], default="rbf")
    C = Float("C", (0.001, 1000), default=1.0, log=True)
    gamma = Float("gamma", (0.0001, 10), default=0.1, log=True)
    degree = Integer("degree", (2, 5), default=3)
    coef0 = Float("coef0", (-1, 1), default=0)

    cs.add_hyperparameters([kernel, C, gamma, degree, coef0])

    # Add forbidden clauses to avoid invalid combinations
    forbidden_clause = ForbiddenAndConjunction(ForbiddenEqualsClause(kernel, "linear"), ForbiddenEqualsClause(degree, 2))

    cs.add_forbidden_clause(forbidden_clause)

    return cs
