from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Creates a ConfigurationSpace for a regression task on a tabular dataset.
    Includes hyperparameters for RandomForestRegressor and GradientBoostingRegressor,
    and the option to use XGBoost with its specific regularization parameters.
    """
    cs = ConfigurationSpace()

    # Define hyperparameters
    model_type = CategoricalHyperparameter("model_type", choices=["random_forest", "gradient_boosting", "xgboost"], default_value="random_forest")
    cs.add_hyperparameter(model_type)

    n_estimators = UniformIntegerHyperparameter("n_estimators", lower=50, upper=200, default_value=100)
    cs.add_hyperparameter(n_estimators)

    max_depth = UniformIntegerHyperparameter("max_depth", lower=3, upper=10, default_value=5)
    cs.add_hyperparameter(max_depth)

    # Random Forest Specific Hyperparameters
    min_samples_split = UniformIntegerHyperparameter("min_samples_split", lower=2, upper=10, default_value=2)
    cs.add_hyperparameter(min_samples_split)

    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", lower=1, upper=10, default_value=1)
    cs.add_hyperparameter(min_samples_leaf)

    # Gradient Boosting Specific Hyperparameters
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=1e-4, upper=1e-1, default_value=1e-2, log=True)
    cs.add_hyperparameter(learning_rate)

    # XGBoost Specific Hyperparameters
    reg_alpha = UniformFloatHyperparameter("reg_alpha", lower=1e-8, upper=1.0, default_value=1e-8, log=True)
    cs.add_hyperparameter(reg_alpha)

    reg_lambda = UniformFloatHyperparameter("reg_lambda", lower=1e-8, upper=1.0, default_value=1e-8, log=True)
    cs.add_hyperparameter(reg_lambda)

    # Define conditions
    from ConfigSpace.conditions import InCondition

    random_forest_condition = InCondition(child=min_samples_split, parent=model_type, values=["random_forest"])
    cs.add_condition(random_forest_condition)

    random_forest_condition_leaf = InCondition(child=min_samples_leaf, parent=model_type, values=["random_forest"])
    cs.add_condition(random_forest_condition_leaf)

    gradient_boosting_condition = InCondition(child=learning_rate, parent=model_type, values=["gradient_boosting"])
    cs.add_condition(gradient_boosting_condition)

    xgboost_condition_alpha = InCondition(child=reg_alpha, parent=model_type, values=["xgboost"])
    cs.add_condition(xgboost_condition_alpha)

    xgboost_condition_lambda = InCondition(child=reg_lambda, parent=model_type, values=["xgboost"])
    cs.add_condition(xgboost_condition_lambda)

    return cs
