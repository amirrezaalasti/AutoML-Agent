from scripts.LLMClient import LLMClient
from typing import Any
from scripts.utils import (
    describe_dataset,
    save_code,
    format_dataset,
    extract_code_block,
    generate_general_dataset_task_types,
)
from scripts.Logger import Logger
from scripts.OpenMLRAG import OpenMLRAG
from configs.api_keys import OPENML_API_KEY, CSV_PATH
from scripts.LLMPlanner import LLMPlanner
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac import Scenario
import os
import json
import time


class AutoMLAgent:
    MAX_RETRIES_PROMPT = 10
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

        self.openML = OpenMLRAG(openml_api_key=OPENML_API_KEY, metafeatures_csv_path=CSV_PATH)
        self.LLMInstructor = LLMPlanner(
            dataset_name=self.dataset_name,
            dataset_description=self.dataset_description,
            dataset_type=self.dataset_type,
        )

    def generate_components(self):
        """
        Generate the configuration space, scenario, and training function code using LLM.
        :return: Tuple containing the generated configuration space, scenario, training function code, last loss, and prompts.
        """
        config_space_suggested_parameters = self.openML.extract_suggested_config_space_parameters(
            dataset_name_in_openml=self.dataset_name,
        )
        instructor_response = self.LLMInstructor.generate_instructions(
            config_space_suggested_parameters,
        )

        self.ui_agent.subheader("Instructor Response")
        # configuration plan:
        self.ui_agent.subheader("Configuration Plan")
        self.ui_agent.write(instructor_response.recommended_configuration)
        # scenario plan:
        self.ui_agent.subheader("Scenario Plan")
        self.ui_agent.write(instructor_response.scenario_plan)
        # train function plan:
        self.ui_agent.subheader("Train Function Plan")
        self.ui_agent.write(instructor_response.train_function_plan)

        # Configuration Space
        self.prompts["config"] = self._create_config_prompt(instructor_response.recommended_configuration)
        self.config_code = self.llm.generate(self.prompts["config"])
        self.logger.log_prompt(self.prompts["config"], {"component": "config"})
        self.logger.log_response(self.config_code, {"component": "config"})
        self._run_generated("config")
        self.ui_agent.subheader("Generated Configuration Space Code")
        self.ui_agent.code(self.config_space, language="python")

        # Scenario
        self.prompts["scenario"] = self._create_scenario_prompt(instructor_response.scenario_plan)
        self.scenario_code = self.llm.generate(self.prompts["scenario"])
        self.logger.log_prompt(self.prompts["scenario"], {"component": "scenario"})
        self.logger.log_response(self.scenario_code, {"component": "scenario"})
        self._run_generated("scenario")
        self.ui_agent.subheader("Generated Scenario Code")
        self.ui_agent.code(self.scenario, language="python")

        # Training Function
        train_plan = instructor_response.train_function_plan
        self.prompts["train_function"] = self._create_train_prompt(train_plan)
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

        self.ui_agent.success("AutoML Agent setup complete!")
        self.ui_agent.subheader("Loss Value")
        self.ui_agent.write(self.last_loss)

        self.ui_agent.subheader("Starting Optimization Process")
        self.ui_agent.write("Starting Optimization Process")

        self.run_scenario(self.scenario_obj, train, self.dataset, self.config_space_obj)

        # Return results and last training loss
        return (
            self.config_space,
            self.scenario,
            self.train_function,
            self.last_loss,
            self.prompts,
            self.logger.experiment_dir,
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

    def _create_config_prompt(self, recommended_configuration) -> str:
        """
        Create a prompt for generating the configuration space code.
        :param config_space_suggested_parameters: Suggested parameters for the configuration space.
        :return: A formatted prompt string for the LLM.
        """
        with open("templates/config_prompt.txt", "r") as f:
            return f.read().format(
                dataset_description=self.dataset_description,
                recommended_configuration=recommended_configuration,
            )

    def _create_scenario_prompt(self, scenario_plan) -> str:
        """
        Create a prompt for generating the scenario code.
        :return: A formatted prompt string for the LLM.
        """

        with open("templates/scenario_prompt.txt", "r") as f:
            base_prompt = f.read().format(
                dataset_description=self.dataset_description,
                scenario_plan=scenario_plan,
                output_directory=self.logger.experiment_dir,
            )

        return base_prompt

    def _create_train_prompt(self, train_function_instructions) -> str:
        """
        Create a prompt for generating the training function code.
        :param train_function_instructions: Instructions for the training function.
        :return: A formatted prompt string for the LLM.
        """

        with open("templates/train_prompt.txt", "r") as f:
            prompt = f.read().format(
                dataset_description=self.dataset_description,
                config_space=self.config_space or "",
                scenario=self.scenario or "",
            )
        # Add specific instructions for the training function
        print("Adding training function instructions")
        print(train_function_instructions)
        prompt += train_function_instructions
        return prompt

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
        incumbent = smac.optimize()

        default_cost = smac.validate(self.config_space_obj.get_default_configuration())
