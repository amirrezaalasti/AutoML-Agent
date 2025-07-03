import openml
import pandas as pd
from autogluon.tabular import TabularPredictor
from scripts.utils import split_dataset_kfold
from numpy.random import RandomState
# 1. Download the "credit-g" dataset from OpenML
dataset = openml.datasets.get_dataset(31)
df: pd.DataFrame = pd.DataFrame(dataset.get_data()[0])

# Reanme class to target
df.rename(columns={"class": "target"}, inplace=True)

# Split dataset into training and testing sets
X = df.drop(columns=["target"])
y = df[["target"]]

X_train, y_train, X_test, y_test = split_dataset_kfold(X =X, y =y, n_folds=5, fold=0, rng=RandomState(42))

autogluon_train_data = pd.concat([X_train, y_train], axis=1)
autogluon_train_data["target"] = y_train.astype(str)
autogluon_test_data = pd.concat([X_test, y_test], axis=1)
autogluon_test_data["target"] = y_test.astype(str)
predictor = TabularPredictor(label="target", log_to_file=True, log_file_path="dummy_log").fit(autogluon_train_data, presets="best_quality",  ag_args_fit={'num_gpus': 1}, time_limit=9000)

performance = predictor.evaluate(autogluon_test_data)
print(performance)