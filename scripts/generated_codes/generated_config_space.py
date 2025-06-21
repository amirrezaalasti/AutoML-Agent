from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter


def get_configspace() -> ConfigurationSpace:
    """
    Creates a ConfigurationSpace for a machine learning pipeline suitable for the given tabular dataset.
    This configuration space includes hyperparameters for data preprocessing (scaling) and a RandomForestClassifier.
    """
    cs = ConfigurationSpace()

    # 1. Data Preprocessing: Scaling
    scaling = CategoricalHyperparameter("scaling", choices=["none", "standard", "minmax"], default_value="standard")
    cs.add_hyperparameter(scaling)

    # 2. Classifier: RandomForestClassifier
    # Add an unparameterized hyperparameter for the classifier choice
    classifier = UnParametrizedHyperparameter("classifier", value="random_forest")
    cs.add_hyperparameter(classifier)

    n_estimators = UniformIntegerHyperparameter("rf_n_estimators", lower=50, upper=500, default_value=100)
    cs.add_hyperparameter(n_estimators)

    max_features = UniformFloatHyperparameter("rf_max_features", lower=0.1, upper=1.0, default_value=0.5)
    cs.add_hyperparameter(max_features)

    min_samples_split = UniformIntegerHyperparameter("rf_min_samples_split", lower=2, upper=20, default_value=2)
    cs.add_hyperparameter(min_samples_split)

    min_samples_leaf = UniformIntegerHyperparameter("rf_min_samples_leaf", lower=1, upper=20, default_value=1)
    cs.add_hyperparameter(min_samples_leaf)

    # Add condition: RandomForest parameters are only relevant if RandomForest is chosen
    # rf_condition = InCondition(child=n_estimators, parent=classifier, values=["random_forest"])
    # cs.add_condition(rf_condition)
    # rf_condition = InCondition(child=max_features, parent=classifier, values=["random_forest"])
    # cs.add_condition(rf_condition)
    # rf_condition = InCondition(child=min_samples_split, parent=classifier, values=["random_forest"])
    # cs.add_condition(rf_condition)
    # rf_condition = InCondition(child=min_samples_leaf, parent=classifier, values=["random_forest"])
    # cs.add_condition(rf_condition)

    return cs
