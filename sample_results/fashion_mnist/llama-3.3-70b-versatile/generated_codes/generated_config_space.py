from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause, EqualsCondition

def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    learning_rate = Categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    eta0 = Float('eta0', bounds=(0.01, 1.0), default=0.1, log=True)
    max_iter = Integer('max_iter', bounds=(100, 1000), default=200)
    tol = Float('tol', bounds=(1e-5, 1e-1), default=1e-3, log=True)
    early_stopping = Categorical('early_stopping', ['True', 'False'], default='False')
    validation_fraction = Float('validation_fraction', bounds=(0.01, 0.5), default=0.1)
    n_jobs = Integer('n_jobs', bounds=(1, 10), default=1)
    random_state = Integer('random_state', bounds=(0, 100), default=42)

    cs.add_hyperparameters([learning_rate, eta0, max_iter, tol, early_stopping, validation_fraction, n_jobs, random_state])

    cond_eta0 = EqualsCondition(eta0, learning_rate, 'constant')
    cs.add_condition(cond_eta0)

    forbidden_eta0_and_early_stopping = ForbiddenAndConjunction(
        ForbiddenEqualsClause(eta0, 0.1),
        ForbiddenEqualsClause(early_stopping, 'True')
    )
    cs.add_forbidden_clause(forbidden_eta0_and_early_stopping)

    return cs
