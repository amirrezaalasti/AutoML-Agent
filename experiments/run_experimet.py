import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_experimenter.experimenter import PyExperimenter
from scripts.AutoMLAgent import AutoMLAgent
from sklearn.datasets import fetch_openml
from config.api_keys import LOCAL_LLAMA_API_KEY
from config.urls import BASE_URL
from py_experimenter.result_processor import ResultProcessor
from typing import Any


class CLIUI:
    """Simple CLI UI agent for batch processing."""

    def __init__(self):
        self.current_dataset = None

    def subheader(self, text: str):
        """Display a subheader."""
        print(f"\n{'='*50}")
        print(f" {text}")
        print(f"{'='*50}")

    def write(self, text: Any):
        """Display text."""
        print(f"  {text}")

    def code(self, code: str, language: str = "python"):
        """Display code."""
        print(f"  [Code - {language}]:")
        print(f"  {code[:200]}..." if len(code) > 200 else f"  {code}")

    def success(self, message: str):
        """Display success message."""
        print(f"  ✅ {message}")

    def error(self, message: str):
        """Display error message."""
        print(f"  ❌ {message}")

    def warning(self, message: str):
        """Display warning message."""
        print(f"  ⚠️  {message}")

    def info(self, message: str):
        """Display info message."""
        print(f"  ℹ️  {message}")


class AutoMLAgentExperimenter:
    def __init__(self):
        self.experimenter = PyExperimenter(
            experiment_configuration_file_path="config/experimenter-config.yml",
            database_credential_file_path="config/database_credentials.yml",
            use_ssh_tunnel=False,
            use_codecarbon=False,
            name="AutoMLAgentRuns",
        )

    def run_automl_experiment(self, parameters: dict, result_processor: ResultProcessor, custom_config: dict):
        """
        Function to be executed by PyExperimenter.
        :param parameters: Dictionary of parameters for the experiment.
        :param result_processor: ResultProcessor object for processing results.
        :param custom_config: Custom configuration for the experiment.
        """
        dataset_name = parameters["dataset_name"]
        dataset_type = parameters["dataset_type"]
        task_type = parameters["task_type"]
        llm_model = parameters["llm_model"]
        dataset_openml_id = parameters["dataset_openml_id"]
        ui_agent = CLIUI()
        n_folds = parameters["n_folds"]
        fold = parameters["fold"]
        X, y = fetch_openml(data_id=dataset_openml_id, return_X_y=True)
        dataset = {"X": X, "y": y}
        agent = AutoMLAgent(
            dataset=dataset,
            dataset_type=dataset_type,
            task_type=task_type,
            ui_agent=ui_agent,
            model_name=llm_model,
            dataset_name=dataset_name,
            api_key=LOCAL_LLAMA_API_KEY,
            base_url=BASE_URL,
            n_folds=n_folds,
            fold=fold,
        )

        config_code, scenario_code, train_function_code, last_loss, metrics, prompts, experiment_dir = agent.generate_components()

        result_processor.process_results(
            {
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_roc_auc": metrics["roc_auc"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "log_dir": experiment_dir,
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
