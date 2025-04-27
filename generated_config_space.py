from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction

def get_configspace():
    cs = ConfigurationSpace(seed=1234)
    learning_rate = Categorical("learning_rate", ["adaptive", "constant"])
    alpha = Float("alpha", bounds=(1e-6, 1e-1), log=True)
    max_iter = Integer("max_iter", bounds=(100, 1000))
    eta0 = Float("eta0", bounds=(1e-4, 1.0), log=True)
    early_stopping = Categorical("early_stopping", [True, False], default=True)

    cs.add([learning_rate, alpha, max_iter, eta0, early_stopping])

    cond = EqualsCondition(eta0, learning_rate, "constant")
    cs.add_condition(cond)

    forbidden_clause = ForbiddenAndConjunction(
        ForbiddenEqualsClause(learning_rate, "adaptive"),
        ForbiddenEqualsClause(eta0, eta0.default_value)
    )
    cs.add_forbidden_clause(forbidden_clause)

    return cs
