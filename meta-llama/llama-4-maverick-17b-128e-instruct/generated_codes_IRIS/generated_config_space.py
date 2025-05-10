from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, EqualsCondition, InCondition, AndConjunction

def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    classifier = Categorical('classifier', ['svm', 'rf', 'knn'])
    cs.add_hyperparameter(classifier)

    C = Float('C', (1e-5,1e5), default=1, log=True)
    cs.add_hyperparameter(C)
    kernel = Categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    cs.add_hyperparameter(kernel)
    degree = Integer('degree', (2,5), default=3)
    cs.add_hyperparameter(degree)
    gamma = Float('gamma', (1e-5,1e2), default=1, log=True)
    cs.add_hyperparameter(gamma)
    coef0 = Float('coef0', (0,1), default=0)
    cs.add_hyperparameter(coef0)

    n_estimators = Integer('n_estimators', (10,1000), default=100)
    cs.add_hyperparameter(n_estimators)
    max_depth = Integer('max_depth', (1,10), default=5) 
    cs.add_hyperparameter(max_depth)

    n_neighbors = Integer('n_neighbors', (1,100), default=5)
    cs.add_hyperparameter(n_neighbors)
    weights = Categorical('weights', ['uniform', 'distance'])
    cs.add_hyperparameter(weights)

    cond_1 = EqualsCondition(C, classifier, 'svm')
    cond_2 = EqualsCondition(kernel, classifier, 'svm')
    cond_6 = EqualsCondition(n_estimators, classifier, 'rf')
    cond_7 = EqualsCondition(max_depth, classifier, 'rf')
    cond_8 = EqualsCondition(n_neighbors, classifier, 'knn')
    cond_9 = EqualsCondition(weights, classifier, 'knn')

    cond_degree = AndConjunction(EqualsCondition(degree, classifier, 'svm'), InCondition(degree, kernel, ['poly']))
    cond_coef0 = AndConjunction(EqualsCondition(coef0, classifier, 'svm'), InCondition(coef0, kernel, ['poly']))
    cond_gamma = AndConjunction(EqualsCondition(gamma, classifier, 'svm'), InCondition(gamma, kernel, ['rbf', 'poly', 'sigmoid']))

    cs.add_condition(cond_1)
    cs.add_condition(cond_2)
    cs.add_condition(cond_6)
    cs.add_condition(cond_7)
    cs.add_condition(cond_8)
    cs.add_condition(cond_9)
    cs.add_condition(cond_degree)
    cs.add_condition(cond_coef0)
    cs.add_condition(cond_gamma)

    return cs
