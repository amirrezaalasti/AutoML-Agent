import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_experimenter.experimenter import PyExperimenter
from scripts.AutoMLAgent import AutoMLAgent
from sklearn.datasets import fetch_openml
from config.api_keys import LOCAL_LLAMA_API_KEY, GOOGLE_API_KEY
from config.urls import BASE_URL
from py_experimenter.result_processor import ResultProcessor
from scripts.CLIUI import CLIUI
from utils.file_operations import load_kaggle_dataset


class AutoMLAgentExperimenter:
    def __init__(self):
        self.experimenter = PyExperimenter(
            experiment_configuration_file_path="/mnt/home/aalasti/agent-smac-project/autoML-Agent/config/experimenter-config.yml",
            database_credential_file_path="/mnt/home/aalasti/agent-smac-project/autoML-Agent/config/database_credentials.yml",
            use_ssh_tunnel=False,
            use_codecarbon=False,
            name="AutoMLAgentRuns",
        )

    def run_automl_experiment(
        self, parameters: dict, result_processor: ResultProcessor, custom_config: dict
    ):
        """
        Function to be executed by PyExperimenter.
        :param parameters: Dictionary of parameters for the experiment.
        :param result_processor: ResultProcessor object for processing results.
        :param custom_config: Custom configuration for the experiment.
        """
        dataset_type = parameters["dataset_type"]
        task_type = parameters["task_type"]
        llm_model = parameters["llm_model"]
        dataset_openml_id = parameters["dataset_openml_id"]
        # Build dataset_name dynamically depending on origin
        if parameters["dataset_origin"] == "kaggle":
            kaggle_dataset_id = parameters.get("kaggle_dataset_id", "")
            dataset_name = (
                parameters["dataset_name"]
                if parameters.get("dataset_name")
                else kaggle_dataset_id
            )
        else:
            dataset_name = parameters["dataset_name"] + "_" + str(dataset_openml_id)
        dataset_origin = parameters["dataset_origin"]
        ui_agent = CLIUI()
        n_folds = parameters["n_folds"]
        fold = parameters["fold"]
        time_budget = parameters.get("time_budget", 86400)
        dataset_folder = "./agent_smac_logs_v2/logs" + "_" + dataset_name

        print(
            f"Experiment configuration: Dataset={dataset_name}, Time Budget={time_budget}s"
        )
        api_key = LOCAL_LLAMA_API_KEY
        base_url = BASE_URL

        if llm_model == "gemini-2.0-flash":
            api_key = GOOGLE_API_KEY
            base_url = None
        if dataset_origin == "openml":
            X, y = fetch_openml(data_id=dataset_openml_id, return_X_y=True)
            dataset = {"X": X, "y": y}
        elif dataset_origin == "kaggle":
            kaggle_dataset_id = parameters.get("kaggle_dataset_id")
            kaggle_file_path = parameters.get("kaggle_file_path")
            kaggle_target_column = parameters.get("kaggle_target_column")

            if (
                not kaggle_dataset_id
                or not kaggle_file_path
                or not kaggle_target_column
            ):
                raise ValueError(
                    "For kaggle origin, provide kaggle_dataset_id, kaggle_file_path, kaggle_target_column in config"
                )

            dataset_df = load_kaggle_dataset(
                dataset_id=kaggle_dataset_id,
                file_path=kaggle_file_path,
            )
            X = dataset_df.drop(columns=[kaggle_target_column])
            y = dataset_df[kaggle_target_column]
            dataset = {"X": X, "y": y}
        agent = AutoMLAgent(
            dataset=dataset,
            dataset_type=dataset_type,
            task_type=task_type,
            ui_agent=ui_agent,
            model_name=llm_model,
            dataset_name=dataset_name,
            api_key=api_key,
            base_url=base_url,
            n_folds=n_folds,
            fold=fold,
            time_budget=time_budget,
            folder_name=dataset_folder,
        )

        try:
            (
                config_code,
                scenario_code,
                train_function_code,
                last_loss,
                metrics,
                prompts,
                experiment_dir,
            ) = agent.generate_components()

            result_processor.process_results(
                {
                    "test_accuracy": metrics.get("accuracy"),
                    "test_f1": metrics.get("f1"),
                    "test_precision": metrics.get("precision"),
                    "test_recall": metrics.get("recall"),
                    "test_roc_auc": metrics.get("roc_auc"),
                    "test_balanced_accuracy": metrics.get("balanced_accuracy"),
                    "test_mcc": metrics.get("mcc"),
                    "log_dir": experiment_dir,
                }
            )

        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            # Process results with error information
            result_processor.process_results(
                {
                    "test_accuracy": None,
                    "test_f1": None,
                    "test_precision": None,
                    "test_recall": None,
                    "test_roc_auc": None,
                    "test_balanced_accuracy": None,
                    "test_mcc": None,
                    "log_dir": f"ERROR: {str(e)}",
                }
            )

    def run_experiment(self):
        self.experimenter.execute(
            experiment_function=self.run_automl_experiment,
            random_order=True,
            max_experiments=1,
        )

    def fill_table(self):
        self.experimenter.fill_table_from_config()


if __name__ == "__main__":
    experimenter = AutoMLAgentExperimenter()
    experimenter.fill_table()
    experimenter.run_experiment()
