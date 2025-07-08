from py_experimenter.result_processor import ResultProcessor
from numpy.random import RandomState
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from pathlib import Path

from utils.file_operations import load_dataset, load_image_dataset 
from utils.data_processing import split_dataset_kfold
from py_experimenter.experimenter import PyExperimenter
import pandas as pd


class AutoGluonWrapper:
    def __init__(
        self,
        dataset_origin: str,
        dataset_name: str,
        dataset_type: str,
        n_folds: int,
        fold: int,
        n_gpus: int,
        n_cpus: int,
        n_gb_ram: int,
        time_budget: int,
        result_processor: ResultProcessor,
    ):
        self.dataset_origin = dataset_origin
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.n_folds = n_folds
        self.fold = fold
        self.n_gpus = n_gpus
        self.n_cpus = n_cpus
        self.n_gb_ram = n_gb_ram
        self.time_budget = time_budget
        self.result_processor = result_processor
        self.log_path = Path(f"logs/baselines/autogluon/{dataset_name}/{fold}")

        self.rng = RandomState(fold)

        self.log_path.mkdir(parents=True, exist_ok=True)

    def get_train_and_test_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        predictor = self.get_predictor()

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        if isinstance(predictor, TabularPredictor):
            predictor = predictor.fit(train_data=train_data, presets="best_quality", num_gpus=self.n_gpus, num_cpus=self.n_cpus, time_limit=self.time_budget)
        elif isinstance(predictor, MultiModalPredictor):    
            predictor.set_num_gpus(self.n_gpus)
            predictor = predictor.fit(train_data=train_data, presets="best_quality", time_limit=self.time_budget)
        else:
            raise ValueError(f"Unsupported predictor type: {type(predictor)}")

        results = predictor.evaluate(test_data)
        return results

    def log_final_results(self, results: dict[str, float | str]):
        results = {f"test_{key}": float(value) for key, value in results.items()}
        self.result_processor.process_results(results)

    def get_predictor(self) -> TabularPredictor | MultiModalPredictor:
        if self.dataset_type == "tabular":
            return TabularPredictor(
                label="target", log_file_path=str(self.log_path / "autogluon.log")
            )

        elif self.dataset_type == "image":
            return MultiModalPredictor(
                label="target",
                path=str(
                    self.log_path
                    / "autogluon"
                    / "multimodal"
                    / "image"
                    / self.dataset_origin
                    / self.dataset_name
                    / str(self.fold)
                )
            )
        else:
            raise ValueError(f"Dataset type {self.dataset_type} not supported")

    def create_dataset(self):
        if self.dataset_type == "tabular":
            X, y = load_dataset(
                dataset_origin=self.dataset_origin, dataset_id=self.dataset_name, 
            )
        elif self.dataset_type == "image":
            X, y = load_image_dataset(
                dataset_origin=self.dataset_origin, dataset_id=self.dataset_name, overwrite=True
            )

        X_train, y_train, X_test, y_test = split_dataset_kfold(
            X=X, y=y, n_folds=self.n_folds, fold=self.fold, rng=self.rng
        )
        return X_train, y_train, X_test, y_test


def run_ml(
    parameters: dict,
    result_processor: ResultProcessor,
    custom_config: dict,
):
    predictor = AutoGluonWrapper(
        dataset_origin=parameters["dataset_origin"],
        dataset_name=parameters["dataset_name"],
        dataset_type=parameters["dataset_type"],
        n_folds=parameters["n_folds"],
        fold=parameters["fold"],
        n_gpus=parameters["n_gpus"],
        n_cpus=parameters["n_cpus"],
        n_gb_ram=parameters["n_gb_ram"],
        time_budget=parameters["time_budget"],
        result_processor=result_processor,
    )
    X_train, y_train, X_test, y_test = predictor.create_dataset()
    results = predictor.run(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    predictor.log_final_results(results)


def fill_table(experimenter: PyExperimenter):
    experimenter.fill_table_from_config()


if __name__ == "__main__":
    experimenter = PyExperimenter(
        experiment_configuration_file_path="baselines/autogluon/autogluon_config.yml",
        database_credential_file_path="config/database_credentials.yml",
        use_ssh_tunnel=False,
        use_codecarbon=False,
    )
    fill_table(experimenter)
    #experimenter.execute(
    #    experiment_function=run_ml, random_order=True, max_experiments=1
    #)
