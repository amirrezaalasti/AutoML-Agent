from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, Constant, EqualsCondition, ForbiddenAndConjunction, ForbiddenEqualsClause

def get_configspace():
    cs = ConfigurationSpace(seed=1234)
    
    learning_rate = Categorical("learning_rate", ["adaptive", "constant"])
    alpha = Float("alpha", [1e-6, 1e-1], log=True)
    max_iter = Integer("max_iter", [100, 1000])
    eta0 = Float("eta0", [1e-4, 1.0], log=True)
    early_stopping = Categorical("early_stopping", [True, False], default=True)
    
    cs.add_hyperparameters([learning_rate, alpha, max_iter, eta0, early_stopping])
    
    cond_eta0 = EqualsCondition(eta0, learning_rate, "constant")
    cs.add_condition(cond_eta0)
    
    penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(learning_rate, "constant"),
        ForbiddenEqualsClause(early_stopping, False)
    )
    cs.add_forbidden_clause(penalty_and_loss)
    
    return cs
