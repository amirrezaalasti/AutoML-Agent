from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction


def get_configspace() -> ConfigurationSpace:
    """
    Returns a ConfigurationSpace object for hyperparameter optimization.

    This configuration space is designed for a classification task on a tabular dataset
    with 120 samples and 4 numerical features. The configuration space includes
    options for data preprocessing, feature engineering, and model-specific
    hyperparameters for a Support Vector Classifier (SVC) and a Random Forest.

    Data preprocessing includes scaling options. Feature engineering consists of
    polynomial feature expansion (degree configurable). The model choices are
    SVC with configurable kernel, C, gamma, and Random Forest with configurable
    number of estimators, max depth, and min samples leaf.
    """
    cs = ConfigurationSpace()

    # 1. Data Preprocessing
    scaler = CategoricalHyperparameter("scaler", choices=["StandardScaler", "MinMaxScaler", "None"], default_value="StandardScaler")
    cs.add_hyperparameter(scaler)

    # 2. Feature Engineering
    use_feature_engineering = CategoricalHyperparameter("use_feature_engineering", choices=[True, False], default_value=False)
    cs.add_hyperparameter(use_feature_engineering)

    poly_degree = UniformIntegerHyperparameter("poly_degree", lower=2, upper=3)
    cs.add_hyperparameter(poly_degree)

    cs.add_condition(InCondition(child=poly_degree, parent=use_feature_engineering, values=[True]))

    # 3. Model Selection
    model_type = CategoricalHyperparameter("model_type", choices=["SVC", "RandomForest"], default_value="SVC")
    cs.add_hyperparameter(model_type)

    # 4. SVC Hyperparameters
    svc_kernel = CategoricalHyperparameter("svc_kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf")
    svc_C = UniformFloatHyperparameter("svc_C", lower=0.001, upper=1000, default_value=1.0, log=True)
    svc_gamma = UniformFloatHyperparameter("svc_gamma", lower=0.0001, upper=10, default_value=0.1, log=True)
    svc_degree = UniformIntegerHyperparameter("svc_degree", lower=2, upper=5)  # Only used for polynomial kernel

    cs.add_hyperparameters([svc_kernel, svc_C, svc_gamma, svc_degree])

    # Conditions for SVC hyperparameters
    kernel_condition = InCondition(child=svc_kernel, parent=model_type, values=["SVC"])
    c_condition = InCondition(child=svc_C, parent=model_type, values=["SVC"])
    gamma_condition = InCondition(child=svc_gamma, parent=model_type, values=["SVC"])
    degree_condition = InCondition(child=svc_degree, parent=svc_kernel, values=["poly"])  # svc_degree only relevant if kernel is poly

    cs.add_conditions([kernel_condition, c_condition, gamma_condition, degree_condition])

    # 5. RandomForest Hyperparameters
    rf_n_estimators = UniformIntegerHyperparameter("rf_n_estimators", lower=10, upper=200)
    rf_max_depth = UniformIntegerHyperparameter("rf_max_depth", lower=2, upper=10)  # Can be None, handled in code
    rf_min_samples_leaf = UniformIntegerHyperparameter("rf_min_samples_leaf", lower=1, upper=10)

    cs.add_hyperparameters([rf_n_estimators, rf_max_depth, rf_min_samples_leaf])

    # Conditions for RandomForest hyperparameters
    n_estimators_condition = InCondition(child=rf_n_estimators, parent=model_type, values=["RandomForest"])
    max_depth_condition = InCondition(child=rf_max_depth, parent=model_type, values=["RandomForest"])
    min_samples_leaf_condition = InCondition(child=rf_min_samples_leaf, parent=model_type, values=["RandomForest"])

    cs.add_conditions([n_estimators_condition, max_depth_condition, min_samples_leaf_condition])

    return cs
