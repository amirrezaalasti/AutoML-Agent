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
dataset = {
    "X": df.drop(columns=["target"]),
    "y": df["target"]
}

train_data, test_data = split_dataset_kfold(dataset, n_folds=5, fold=0, rng=RandomState(42))

autogluon_train_data = train_data["X"]
autogluon_train_data["target"] = train_data["y"].astype(str)
autogluon_test_data = test_data["X"]
autogluon_test_data["target"] = test_data["y"].astype(str)
predictor = TabularPredictor(label="target", log_to_file=True, log_file_path="dummy_log").fit(autogluon_train_data, presets="best_quality_hpo",  ag_args_fit={'num_gpus': 1}, time_limit=9000)

performance = predictor.evaluate(autogluon_test_data)
print(performance)