from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, EqualsCondition, ForbiddenEqualsClause, ForbiddenAndConjunction

def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    model = Categorical('model', ['mlp', 'cnn'], default='mlp')
    cs.add_hyperparameter(model)

    learning_rate = Categorical('learning_rate', ['constant', 'invscaling', 'adaptive'], default='constant')
    cs.add_hyperparameter(learning_rate)

    eta0 = Float('eta0', bounds=(1e-7, 1e-1), default=1e-3, log=True)
    cs.add_hyperparameter(eta0)
    cond_eta0 = EqualsCondition(eta0, learning_rate, 'constant')
    cs.add_condition(cond_eta0)

    max_iter = Integer('max_iter', bounds=(100, 1000), default=500)
    cs.add_hyperparameter(max_iter)

    penalty = Categorical('penalty', ['l1', 'l2'], default='l2')
    cs.add_hyperparameter(penalty)

    loss = Categorical('loss', ['hinge', 'log_loss'], default='log_loss')
    cs.add_hyperparameter(loss)

    alpha = Float('alpha', bounds=(1e-6, 1e-2), default=1e-4, log=True)
    cs.add_hyperparameter(alpha)

    hidden_layer_sizes = Integer('hidden_layer_sizes', bounds=(50, 200), default=100)
    cs.add_hyperparameter(hidden_layer_sizes)

    forbidden_clause_1 = ForbiddenAndConjunction(
        ForbiddenEqualsClause(penalty, 'l1'),
        ForbiddenEqualsClause(loss, 'hinge')
    )
    cs.add_forbidden_clause(forbidden_clause_1)

    return cs
