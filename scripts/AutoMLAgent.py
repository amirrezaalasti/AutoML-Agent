from scripts.LLMClient import LLMClient
from typing import Any
from scripts.utils import (
    describe_dataset,
    save_code,
    format_dataset,
    extract_code_block,
)
from scripts.Logger import Logger
from scripts.OpenMLRAG import OpenMLRAG
from scripts.LLMPlanner import LLMPlanner
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac import Scenario
import os
import json


class AutoMLAgent:
    MAX_RETRIES_PROMPT = 5
    MAX_RETRIES_ABORT = 20

    def __init__(
        self,
        dataset: Any,
        llm_client: LLMClient,
        dataset_type: str,
        ui_agent: Any,
        dataset_name: str = None,
    ):
        """
        Initialize the AutoML Agent with dataset, LLM client, and UI agent.
        :param dataset: The dataset to be used for training.
        :param llm_client: The LLM client for generating code.
        :param dataset_type: The type of dataset (e.g., "tabular", "image", etc.).
        :param ui_agent: The UI agent for displaying results and prompts.
        :param dataset_name: Optional name of the dataset for logging and display.
        """
        self.dataset = format_dataset(dataset)
        self.dataset_type = dataset_type
        self.dataset_description = describe_dataset(self.dataset, dataset_type)
        self.dataset_name = dataset_name if dataset_name else None

        self.llm = llm_client

        self.config_code = None
        self.scenario_code = None
        self.train_function_code = None

        self.config_space = None
        self.scenario = None
        self.train_function = None

        self.scenario_obj = None
        self.config_space_obj = None

        self.error_counters = {"config": 0, "scenario": 0, "train_function": 0}
        self.error_history = {"config": [], "scenario": [], "train_function": []}
        self.namespace = {}

        # store original prompts for retries
        self.prompts = {}
        self.ui_agent = ui_agent

        # Initialize logger
        self.logger = Logger(
            model_name=llm_client.model_name,
            dataset_name=dataset_name,
            base_log_dir="./logs",
        )

        self.openML = OpenMLRAG()
        self.LLMInstructor = LLMPlanner(
            dataset_name=self.dataset_name,
            dataset_description=self.dataset_description,
            dataset_type=self.dataset_type,
        )

    def extract_suggested_config_space_parameters(self, dataset_name_in_openml: str = None):
        sample_datasets = self.openML.get_related_datasets(dataset_name=dataset_name_in_openml)

        tasks = self.openML.get_related_tasks_of_dataset(dataset_id=sample_datasets.did, task_type="classification")
        task_ids = tasks["tid"].tolist()

        parameters = self.openML.get_setup_parameters_of_tasks(task_ids=task_ids)

        return parameters

    def generate_components(self):
        instructor_response = self.LLMInstructor.generate_instructions()
        config_space_suggested_parameters = self.extract_suggested_config_space_parameters(dataset_name_in_openml=instructor_response.dataset_name)

        # Configuration Space
        self.prompts["config"] = self._create_config_prompt(config_space_suggested_parameters)
        self.config_code = self.llm.generate(self.prompts["config"])
        self.logger.log_prompt(self.prompts["config"], {"component": "config"})
        self.logger.log_response(self.config_code, {"component": "config"})
        self._run_generated("config")
        self.ui_agent.subheader("Generated Configuration Space Code")
        self.ui_agent.code(self.config_space, language="python")

        # Scenario
        self.prompts["scenario"] = self._create_scenario_prompt()
        self.scenario_code = self.llm.generate(self.prompts["scenario"])
        self.logger.log_prompt(self.prompts["scenario"], {"component": "scenario"})
        self.logger.log_response(self.scenario_code, {"component": "scenario"})
        self._run_generated("scenario")
        self.ui_agent.subheader("Generated Scenario Code")
        self.ui_agent.code(self.scenario, language="python")

        # Training Function
        self.prompts["train_function"] = self._create_train_prompt()
        self.train_function_code = self.llm.generate(self.prompts["train_function"])
        self.logger.log_prompt(self.prompts["train_function"], {"component": "train_function"})
        self.logger.log_response(self.train_function_code, {"component": "train_function"})
        self._run_generated("train_function")
        self.ui_agent.subheader("Generated Training Function Code")
        self.ui_agent.code(self.train_function, language="python")

        # Save outputs
        save_code(self.config_space, "scripts/generated_codes/generated_config_space.py")
        save_code(self.scenario, "scripts/generated_codes/generated_scenario.py")
        save_code(self.train_function, "scripts/generated_codes/generated_train_function.py")

        from scripts.generated_codes.generated_train_function import train

        self.run_scenario(self.scenario_obj, train, self.dataset, self.config_space_obj)

        # Return results and last training loss
        return (
            self.config_space,
            self.scenario,
            self.train_function,
            self.last_loss,
            self.prompts,
        )

    def _run_generated(self, component: str):
        """Generic runner for config, scenario, or train_function"""
        retry_count = 0
        while True:
            code_attr = f"{component}_code"
            raw_code = getattr(self, code_attr)
            source = extract_code_block(raw_code)
            self.logger.log_prompt(f"Running {component} code:", {"component": component, "action": "run"})
            self.logger.log_response(source, {"component": component, "action": "run"})

            try:
                if component == "config":
                    exec(source, self.namespace)
                    cfg = self.namespace["get_configspace"]()
                    self.config_space = source
                    self.config_space_obj = cfg
                    self.logger.log_response(
                        f"Configuration space generated successfully",
                        {"component": component, "status": "success"},
                    )

                elif component == "scenario":
                    cfg = self.namespace["get_configspace"]()
                    exec(source, self.namespace)
                    scenario_obj = self.namespace["generate_scenario"](cfg)
                    self.scenario = source
                    self.scenario_obj = scenario_obj
                    self.logger.log_response(
                        f"Scenario generated successfully",
                        {"component": component, "status": "success"},
                    )

                elif component == "train_function":
                    cfg = self.namespace["get_configspace"]()
                    exec(source, self.namespace)
                    train_fn = self.namespace["train"]
                    sampled = cfg.sample_configuration()
                    loss = train_fn(cfg=sampled, dataset=self.dataset, seed=0)
                    self.train_function = source
                    self.last_loss = loss
                    self.logger.log_response(
                        f"Training executed successfully, loss: {loss}",
                        {"component": component, "status": "success", "loss": loss},
                    )

                return  # success

            except Exception as e:
                retry_count += 1
                self.error_counters[component] += 1
                err = str(e)
                # record error history
                self.error_history[component].append(err)
                self.logger.log_error(
                    f"Error in {component} (#{retry_count}): {err}",
                    error_type=f"{component.upper()}_ERROR",
                )

                # Abort after too many retries
                if self.error_counters[component] > self.MAX_RETRIES_ABORT:
                    self.logger.log_error(
                        f"Max abort retries reached for {component}. Using last generated code.",
                        error_type=f"{component.upper()}_ABORT",
                    )
                    if component == "train_function":
                        self.last_loss = getattr(self, "last_loss", None)
                    return

                # Refresh code from original prompt after threshold
                if retry_count == self.MAX_RETRIES_PROMPT:
                    self.logger.log_prompt(
                        f"Retry limit reached for {component}. Fetching fresh code from LLM.",
                        {"component": component, "action": "retry"},
                    )
                    fresh = self.llm.generate(self.prompts[component])
                    setattr(self, code_attr, fresh)
                    retry_count = 0
                else:
                    # Ask LLM to fix errors with history of recent issues
                    recent_errors = "\n".join(self.error_history[component][-self.MAX_RETRIES_PROMPT :])
                    fix_prompt = self._create_fix_prompt(recent_errors, raw_code)
                    self.logger.log_prompt(fix_prompt, {"component": component, "action": "fix"})
                    fixed = self.llm.generate(fix_prompt)
                    self.logger.log_response(fixed, {"component": component, "action": "fix"})
                    setattr(self, code_attr, fixed)

    def _create_config_prompt(self, config_space_suggested_parameters) -> str:
        with open("templates/config_prompt.txt", "r") as f:
            return f.read().format(
                dataset_description=self.dataset_description,
                config_space_suggested_parameters=config_space_suggested_parameters,
            )

    def _create_scenario_prompt(self) -> str:
        with open("templates/scenario_prompt.txt", "r") as f:
            base_prompt = f.read().format(dataset=self.dataset_description)

        # Add documentation context with specific guidance
        if os.path.exists("collected_docs/smac_docs.json"):
            with open("collected_docs/smac_docs.json", "r") as f:
                smac_docs = json.load(f)

            # Add specific guidance about using the documentation
            doc_context = """
            Based on the following SMAC documentation, analyze the dataset characteristics and choose appropriate:
            1. Facade type (e.g., MultiFidelityFacade for multi-fidelity optimization)
            2. Budget settings (min_budget and max_budget)
            3. Number of workers (n_workers)
            4. Other relevant scenario parameters

            SMAC Documentation:
            """
            doc_context += "\n\n".join([doc["content"] for doc in smac_docs])

            # Add specific questions to guide the LLM
            doc_context += """
            Please analyze the dataset and documentation to determine:
            1. Should multi-fidelity optimization be used? (Consider dataset size and training time)
            2. What budget range is appropriate? (Consider training epochs or data subsets)
            3. How many workers should be used? (Consider available resources)
            4. Are there any special considerations for this dataset type?

            Then generate a scenario configuration that best suits this dataset.
            """
            return base_prompt + doc_context

    def _create_train_prompt(self) -> str:
        with open("templates/train_prompt.txt", "r") as f:
            return f.read().format(
                dataset_description=self.dataset_description,
                config_space=self.config_space or "",
                scenario=self.scenario or "",
            )

    def _create_fix_prompt(self, errors: str, code: str) -> str:
        """Include recent error history when asking LLM to fix code"""
        with open("templates/fix_prompt.txt", "r") as f:
            return f.read().format(code=code, errors=errors)

    def run_scenario(self, scenario: Scenario, train_fn: Any, dataset: Any, cfg: Any):
        """Run the scenario code and return the results"""

        def smac_train_function(config, seed: int = 0) -> float:
            """Wrapper function to call the training function with the correct parameters"""
            return train_fn(cfg=config, dataset=dataset, seed=seed)

        smac = HPOFacade(
            scenario,
            smac_train_function,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )
        smac.optimize()
