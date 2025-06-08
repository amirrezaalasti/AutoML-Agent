from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause


def get_configspace():
    cs = ConfigurationSpace()

    # Define hyperparameters
    classifier_type = Categorical("classifier_type", ["SVM", "DecisionTree"], default="SVM")

    # SVM hyperparameters
    svm_kernel = Categorical("svm_kernel", ["linear", "rbf", "poly", "sigmoid"], default="rbf")
    svm_C = Float("svm_C", bounds=(0.001, 1000), default=1.0, log=True)
    svm_gamma = Float("svm_gamma", bounds=(0.0001, 10), default=0.1, log=True)
    svm_degree = Integer("svm_degree", bounds=(2, 5), default=3)

    # Decision Tree hyperparameters
    dt_criterion = Categorical("dt_criterion", ["gini", "entropy"], default="gini")
    dt_max_depth = Integer("dt_max_depth", bounds=(1, 20), default=5)
    dt_min_samples_split = Integer("dt_min_samples_split", bounds=(2, 20), default=2)
    dt_min_samples_leaf = Integer("dt_min_samples_leaf", bounds=(1, 20), default=1)

    # Add hyperparameters to the configuration space
    cs.add([classifier_type, svm_kernel, svm_C, svm_gamma, svm_degree, dt_criterion, dt_max_depth, dt_min_samples_split, dt_min_samples_leaf])

    # Add forbidden clauses
    forbidden_svm = ForbiddenAndConjunction(
        ForbiddenEqualsClause(classifier_type, "SVM"), ForbiddenEqualsClause(dt_criterion, "gini")  # This clause is problematic
    )

    forbidden_dt = ForbiddenAndConjunction(ForbiddenEqualsClause(classifier_type, "DecisionTree"), ForbiddenEqualsClause(svm_kernel, "rbf"))

    # Remove the problematic forbidden clause.
    # cs.add_forbidden_clause(forbidden_svm) # Remove this line

    cs.add_forbidden_clause(forbidden_dt)

    return cs
