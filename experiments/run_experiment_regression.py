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


class AutoMLAgentExperimenter:
    def __init__(self):
        self.experimenter = PyExperimenter(
            experiment_configuration_file_path="/mnt/home/aalasti/agent-smac-project/autoML-Agent/config/experimenter-config-regression.yml",
            database_credential_file_path="/mnt/home/aalasti/agent-smac-project/autoML-Agent/config/database_credentials.yml",
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
        dataset_type = parameters["dataset_type"]
        task_type = parameters["task_type"]
        llm_model = parameters["llm_model"]
        dataset_openml_id = parameters["dataset_openml_id"]
        dataset_name = parameters["dataset_name"] + "_" + str(dataset_openml_id)
        ui_agent = CLIUI()
        n_folds = parameters["n_folds"]
        fold = parameters["fold"]
        time_budget = parameters.get("time_budget", 86400)
        dataset_folder = "./agent_smac_logs/logs" + "_" + dataset_name

        print(f"Experiment configuration: Dataset={dataset_name}, Time Budget={time_budget}s")
        api_key = LOCAL_LLAMA_API_KEY
        base_url = BASE_URL

        if llm_model == "gemini-2.0-flash":
            api_key = GOOGLE_API_KEY
            base_url = None

        X, y = fetch_openml(data_id=dataset_openml_id, return_X_y=True)
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
                    "test_mse": metrics.get("mse"),
                    "test_mae": metrics.get("mae"),
                    "test_r2": metrics.get("r2"),
                    "log_dir": experiment_dir,
                }
            )

        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            # Process results with error information
            result_processor.process_results(
                {
                    "test_mse": None,
                    "test_mae": None,
                    "test_r2": None,
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
