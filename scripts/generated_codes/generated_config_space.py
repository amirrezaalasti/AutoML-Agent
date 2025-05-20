from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause


def get_configspace():
    cs = ConfigurationSpace(seed=1234)

    # Define hyperparameters
    classifier = Categorical("classifier", ["knn", "svm", "random_forest"], default="knn")
    cs.add_hyperparameter(classifier)

    # KNN parameters
    n_neighbors = Integer("n_neighbors", bounds=(1, 20), default=5)
    weights = Categorical("weights", ["uniform", "distance"], default="uniform")
    p = Categorical("p", [1, 2], default=2)
    cs.add_hyperparameters([n_neighbors, weights, p])

    # SVM parameters
    C = Float("C", bounds=(1e-5, 10), default=1.0, log=True)
    kernel = Categorical("kernel", ["linear", "rbf", "poly", "sigmoid"], default="rbf")
    degree = Integer("degree", bounds=(2, 5), default=3)
    gamma = Categorical("gamma", ["scale", "auto"], default="scale")
    cs.add_hyperparameters([C, kernel, degree, gamma])

    # Random Forest parameters
    n_estimators = Integer("n_estimators", bounds=(10, 200), default=100)
    max_depth = Integer("max_depth", bounds=(2, 10), default=None)
    min_samples_split = Integer("min_samples_split", bounds=(2, 10), default=2)
    min_samples_leaf = Integer("min_samples_leaf", bounds=(1, 10), default=1)
    cs.add_hyperparameters([n_estimators, max_depth, min_samples_split, min_samples_leaf])

    # Add forbidden clauses
    forbidden_knn = ForbiddenAndConjunction(ForbiddenEqualsClause(classifier, "knn"), ForbiddenEqualsClause(kernel, "linear"))

    forbidden_svm = ForbiddenAndConjunction(ForbiddenEqualsClause(classifier, "svm"), ForbiddenEqualsClause(n_neighbors, 5))

    forbidden_rf = ForbiddenAndConjunction(ForbiddenEqualsClause(classifier, "random_forest"), ForbiddenEqualsClause(kernel, "rbf"))

    cs.add_forbidden_clause(forbidden_knn)
    cs.add_forbidden_clause(forbidden_svm)
    cs.add_forbidden_clause(forbidden_rf)

    return cs
