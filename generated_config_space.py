from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, Constant, EqualsCondition, ForbiddenAndConjunction, ForbiddenEqualsClause

def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    learning_rate = Categorical('learning_rate', ['constant', 'invscaling', 'adaptive'], default='constant')
    eta0 = Float('eta0', bounds=(1e-7, 1e-1), default=1e-4, log=True)
    power_t = Float('power_t', bounds=(0, 1), default=0.5)

    cs.add([learning_rate, eta0, power_t])

    cs.add_condition(EqualsCondition(eta0, learning_rate, 'constant'))

    penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(learning_rate, "adaptive"),
        ForbiddenEqualsClause(power_t, 0)
    )
    cs.add_forbidden_clause(penalty_and_loss)

    return cs
