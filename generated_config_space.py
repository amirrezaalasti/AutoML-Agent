from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, Constant, EqualsCondition, ForbiddenAndConjunction, ForbiddenEqualsClause

def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    learning_rate = Categorical("learning_rate", ["adaptive", "constant"], default="adaptive")
    alpha = Float("alpha", (1e-6,1e-1), log=True, default=1e-2)
    max_iter = Integer("max_iter", (100,1000), default=500)
    eta0 = Float("eta0", (1e-4,1.0), log=True, default=1e-2)
    early_stopping = Categorical("early_stopping", [True, False], default=True)

    cs.add([learning_rate, alpha, max_iter, early_stopping, eta0])

    cs.add_condition(EqualsCondition(eta0, learning_rate, "constant"))

    penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(learning_rate, "adaptive"),
        ForbiddenEqualsClause(eta0,0.01)
    )
    cs.add_forbidden_clause(penalty_and_loss)

    return cs
