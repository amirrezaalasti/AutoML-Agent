from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, InCondition
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction


def get_configspace() -> ConfigurationSpace:
    """
    Returns a ConfigurationSpace object for hyperparameter optimization.
    """
    cs = ConfigurationSpace()

    # --- Model Choice ---
    model_type = CategoricalHyperparameter("model_type", choices=["logistic_regression", "random_forest", "svm"], default_value="random_forest")
    cs.add_hyperparameter(model_type)

    # --- Logistic Regression Hyperparameters ---
    lr_C = UniformFloatHyperparameter("logistic_regression:C", lower=1e-5, upper=10, default_value=1.0, log=True)
    lr_penalty = CategoricalHyperparameter("logistic_regression:penalty", choices=["l1", "l2"], default_value="l2")

    cs.add_hyperparameters([lr_C, lr_penalty])

    condition_lr = InCondition(child=lr_C, parent=model_type, values=["logistic_regression"])
    condition_penalty = InCondition(child=lr_penalty, parent=model_type, values=["logistic_regression"])

    cs.add_conditions([condition_lr, condition_penalty])

    # --- Random Forest Hyperparameters ---
    rf_n_estimators = UniformIntegerHyperparameter("random_forest:n_estimators", lower=10, upper=200, default_value=100)
    rf_max_depth = UniformIntegerHyperparameter("random_forest:max_depth", lower=2, upper=10, default_value=5)
    rf_min_samples_split = UniformIntegerHyperparameter("random_forest:min_samples_split", lower=2, upper=10, default_value=2)
    rf_min_samples_leaf = UniformIntegerHyperparameter("random_forest:min_samples_leaf", lower=1, upper=10, default_value=1)

    cs.add_hyperparameters([rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_min_samples_leaf])

    condition_rf = InCondition(child=rf_n_estimators, parent=model_type, values=["random_forest"])
    condition_max_depth = InCondition(child=rf_max_depth, parent=model_type, values=["random_forest"])
    condition_min_samples_split = InCondition(child=rf_min_samples_split, parent=model_type, values=["random_forest"])
    condition_min_samples_leaf = InCondition(child=rf_min_samples_leaf, parent=model_type, values=["random_forest"])

    cs.add_conditions([condition_rf, condition_max_depth, condition_min_samples_split, condition_min_samples_leaf])

    # --- SVM Hyperparameters ---
    svm_C = UniformFloatHyperparameter("svm:C", lower=1e-5, upper=10, default_value=1.0, log=True)
    svm_kernel = CategoricalHyperparameter("svm:kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
    svm_gamma = UniformFloatHyperparameter("svm:gamma", lower=1e-5, upper=1, default_value=0.1, log=True)
    svm_degree = UniformIntegerHyperparameter("svm:degree", lower=2, upper=5, default_value=3)

    cs.add_hyperparameters([svm_C, svm_kernel, svm_gamma, svm_degree])

    condition_svm = InCondition(child=svm_C, parent=model_type, values=["svm"])
    condition_kernel = InCondition(child=svm_kernel, parent=model_type, values=["svm"])

    cs.add_conditions([condition_svm, condition_kernel])

    gamma_condition = InCondition(child=svm_gamma, parent=model_type, values=["svm"])
    degree_condition = InCondition(child=svm_degree, parent=model_type, values=["svm"])

    cs.add_conditions([gamma_condition, degree_condition])

    # Fix for "Adding a second parent condition" error
    forbidden_degree = ForbiddenAndConjunction(ForbiddenEqualsClause(svm_kernel, "linear"), ForbiddenEqualsClause(svm_degree, 3))
    cs.add_forbidden_clauses([forbidden_degree])

    forbidden_degree_sig = ForbiddenAndConjunction(ForbiddenEqualsClause(svm_kernel, "sigmoid"), ForbiddenEqualsClause(svm_degree, 3))
    cs.add_forbidden_clauses([forbidden_degree_sig])

    forbidden_gamma = ForbiddenAndConjunction(ForbiddenEqualsClause(svm_kernel, "linear"), ForbiddenEqualsClause(svm_gamma, 0.1))

    cs.add_forbidden_clauses([forbidden_gamma])

    return cs
