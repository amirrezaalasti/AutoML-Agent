from scripts.LLMClient import LLMClient
from typing import Any
from utils import (
    describe_dataset,
    save_code_to_file,
    format_dataset,
    extract_code_block,
    split_dataset_kfold,
    FACADE_PARAMETER_REQUIREMENTS,
)
from scripts.Logger import Logger
from scripts.OpenMLRAG import OpenMLRAG
from config.api_keys import OPENML_API_KEY
from config.configs_path import (
    OPENML_CSV_PATH,
    EXPERIMENTER_CONFIG_PATH,
    EXPERIMENTER_DATABASE_CREDENTIALS_PATH,
)
from scripts.LLMPlanner import LLMPlanner
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac import Scenario
from py_experimenter.experimenter import PyExperimenter
from smac.facade.hyperband_facade import HyperbandFacade as HyperbandFacade
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as MultiFidelityFacade
from smac.facade.blackbox_facade import BlackBoxFacade
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error


class AutoMLAgent:
    MAX_RETRIES_PROMPT = 10
    MAX_RETRIES_ABORT = 20

    def __init__(
        self,
        dataset: Any,
        dataset_type: str,
        ui_agent: Any,
        dataset_name: str = None,
        task_type: str = None,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
    ):
        """
        Initialize the AutoML Agent with dataset, LLM client, and UI agent.
        :param dataset: The dataset to be used for training.
        :param llm_client: The LLM client for generating code.
        :param dataset_type: The type of dataset (e.g., "tabular", "image", etc.).
        :param ui_agent: The UI agent for displaying results and prompts.
        :param dataset_name: Optional name of the dataset for logging and display.
        :param task_type: Optional task type for logging and display.
        """
        self.dataset = format_dataset(dataset)
        self.dataset, self.dataset_test = split_dataset_kfold(self.dataset, n_folds=5, fold=0, rng=np.random.RandomState(42))
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name if dataset_name else None
        self.task_type = task_type if task_type else None
        self.dataset_description = describe_dataset(self.dataset, dataset_type, task_type)

        self.config_code = None
        self.scenario_code = None
        self.train_function_code = None

        self.scenario_obj = None
        self.config_space_obj = None

        self.error_counters = {"config": 0, "scenario": 0, "train_function": 0}
        self.error_history = {"config": [], "scenario": [], "train_function": []}
        self.namespace = {}

        # store original prompts for retries
        self.prompts = {}

        self.llm = LLMClient(api_key=api_key, model_name=model_name, base_url=base_url)
        self.ui_agent = ui_agent

        # Initialize logger
        self.logger = Logger(
            model_name=model_name,
            dataset_name=dataset_name,
            base_log_dir="./logs",
        )

        self.openML = OpenMLRAG(openml_api_key=OPENML_API_KEY, metafeatures_csv_path=OPENML_CSV_PATH)
        self.LLMInstructor = LLMPlanner(
            dataset_name=self.dataset_name,
            dataset_description=self.dataset_description,
            dataset_type=self.dataset_type,
            task_type=self.task_type,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        try:
            # self.experimenter = PyExperimenter(
            #     experiment_configuration_file_path=EXPERIMENTER_CONFIG_PATH,
            #     name="AutoMLAgent",
            #     database_credentials_file_path=EXPERIMENTER_DATABASE_CREDENTIALS_PATH,
            #     experiment_dir=self.logger.experiment_dir,
            #     use_ssh_tunnel=False,
            #     use_codecarbon=False,
            # )
            self.experimenter = None
        except Exception as e:
            self.ui_agent.error(f"Error initializing experimenter: {e}")
            self.experimenter = None

        self.experimenter_row_dict = {
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "task_type": self.task_type,
        }

        self.smac_facades = {
            "HyperbandFacade": HyperbandFacade,
            "MultiFidelityFacade": MultiFidelityFacade,
            "HyperparameterOptimizationFacade": HPOFacade,
            "BlackBoxFacade": BlackBoxFacade,
        }

    def generate_components(self):
        """
        Generate the configuration space, scenario, and training function code using LLM.
        :return: Tuple containing the generated configuration space, scenario, training function code, last loss, and prompts.
        """
        config_space_suggested_parameters = self.openML.extract_suggested_config_space_parameters(
            dataset_name=self.dataset_name,
            X=self.dataset["X"],
            y=self.dataset["y"],
        )
        instructor_response = self.LLMInstructor.generate_instructions(
            config_space_suggested_parameters,
        )
        self.suggested_facade = self.smac_facades[instructor_response.suggested_facade]

        self.ui_agent.subheader("Instructor Response")
        # configuration plan:
        self.ui_agent.subheader("Configuration Plan")
        self.ui_agent.write(instructor_response.configuration_plan)
        # scenario plan:
        self.ui_agent.subheader("Scenario Plan")
        self.ui_agent.write(instructor_response.scenario_plan)
        # train function plan:
        self.ui_agent.subheader("Train Function Plan")
        self.ui_agent.write(instructor_response.train_function_plan)
        # scenario facade:
        self.ui_agent.subheader("Suggested Facade")
        self.ui_agent.write(instructor_response.suggested_facade)

        # Configuration Space
        self.prompts["config"] = self._create_config_prompt(
            instructor_response.configuration_plan,
            # self.task_type,
        )
        self.config_code = self.llm.generate(self.prompts["config"])

        self.logger.log_prompt(self.prompts["config"], {"component": "config"})
        self.logger.log_response(self.config_code, {"component": "config"})
        self._run_generated("config")
        self.ui_agent.subheader("Generated Configuration Space Code")
        self.ui_agent.code(self.config_code, language="python")

        # Scenario
        self.prompts["scenario"] = self._create_scenario_prompt(instructor_response.scenario_plan, instructor_response.suggested_facade)
        self.scenario_code = self.llm.generate(self.prompts["scenario"])
        self.logger.log_prompt(self.prompts["scenario"], {"component": "scenario"})
        self.logger.log_response(self.scenario_code, {"component": "scenario"})
        self._run_generated("scenario")
        self.ui_agent.subheader("Generated Scenario Code")
        self.ui_agent.code(self.scenario_code, language="python")

        # Training Function
        train_plan = instructor_response.train_function_plan
        self.prompts["train_function"] = self._create_train_prompt(train_plan, instructor_response.suggested_facade)
        self.train_function_code = self.llm.generate(self.prompts["train_function"])
        self.logger.log_prompt(self.prompts["train_function"], {"component": "train_function"})
        self.logger.log_response(self.train_function_code, {"component": "train_function"})
        self._run_generated("train_function")
        self.ui_agent.subheader("Generated Training Function Code")
        self.ui_agent.code(self.train_function_code, language="python")

        # Save outputs
        save_code_to_file(self.config_code, "scripts/generated_codes/generated_config_space.py")
        save_code_to_file(self.scenario_code, "scripts/generated_codes/generated_scenario.py")
        save_code_to_file(
            self.train_function_code,
            "scripts/generated_codes/generated_train_function.py",
        )

        # Use the train function that was just generated and tested, not the imported one
        # This ensures consistency between the tested function and the one used in SMAC
        train_fn = self.namespace["train"]

        self.ui_agent.success("AutoML Agent setup complete!")
        self.ui_agent.subheader("Loss Value")
        self.ui_agent.write(self.last_loss)

        self.ui_agent.subheader("Starting Optimization Process")
        self.ui_agent.write("Starting Optimization Process")

        self.run_scenario(self.scenario_obj, train_fn, self.dataset, self.config_space_obj)
        if self.experimenter:
            self.experimenter.fill_table_with_rows(rows=[self.experimenter_row_dict])

        # Return results and last training loss
        return (
            self.config_code,
            self.scenario_code,
            self.train_function_code,
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
                    self.config_code = source
                    cfg = self.namespace["get_configspace"]()
                    self.config_space_obj = cfg
                    self.logger.log_response(
                        f"Configuration space generated successfully",
                        {"component": component, "status": "success"},
                    )

                elif component == "scenario":
                    cfg = self.namespace["get_configspace"]()
                    exec(source, self.namespace)
                    self.scenario_code = source
                    scenario_obj = self.namespace["generate_scenario"](cfg)
                    self.scenario_obj = scenario_obj
                    self.logger.log_response(
                        f"Scenario generated successfully",
                        {"component": component, "status": "success"},
                    )

                elif component == "train_function":
                    cfg = self.namespace["get_configspace"]()
                    exec(source, self.namespace)
                    self.train_function_code = source
                    train_fn = self.namespace["train"]
                    sampled = cfg.sample_configuration()

                    # Test train function with budget parameter for compatibility
                    # Try with budget first (for multi-fidelity facades), then without
                    try:
                        result = train_fn(cfg=sampled, dataset=self.dataset, seed=0, budget=None)
                    except TypeError:
                        # If budget parameter not accepted, try without it
                        result = train_fn(cfg=sampled, dataset=self.dataset, seed=0)

                    # Handle both tuple and single value returns
                    if isinstance(result, tuple):
                        loss, model = result
                        self.last_loss = loss
                        self.last_model = model
                    else:
                        loss = result
                        self.last_loss = loss
                        self.last_model = None
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

    def _create_scenario_prompt(self, scenario_plan, suggested_facade) -> str:
        """
        Create a prompt for generating the scenario code.
        :return: A formatted prompt string for the LLM.
        """

        with open("templates/scenario_prompt.txt", "r") as f:
            base_prompt = f.read().format(
                dataset_description=self.dataset_description,
                scenario_plan=scenario_plan,
                output_directory=self.logger.experiment_dir,
                suggested_facade=suggested_facade,
            )

        return base_prompt

    def _create_train_prompt(self, train_function_instructions, suggested_facade) -> str:
        """
        Create a prompt for generating the training function code.
        :param train_function_instructions: Instructions for the training function.
        :param suggested_facade: The suggested SMAC facade.
        :return: A formatted prompt string for the LLM.
        """

        with open("templates/train_prompt.txt", "r") as f:
            prompt = f.read().format(
                dataset_description=self.dataset_description,
                config_space=self.config_code or "",
                scenario=self.scenario_code or "",
                train_function_plan=train_function_instructions,
                suggested_facade=suggested_facade,
            )

        return prompt

    def _create_fix_prompt(self, errors: str, code: str) -> str:
        """Include recent error history when asking LLM to fix code"""
        with open("templates/fix_prompt.txt", "r") as f:
            return f.read().format(code=code, errors=errors)

    def run_scenario(self, scenario: Scenario, train_fn: Any, dataset: Any, cfg: Any):
        """Run the scenario code and return the results"""

        # Get the facade name to determine parameter requirements
        facade_name = self.suggested_facade.__name__
        required_params = FACADE_PARAMETER_REQUIREMENTS.get(facade_name, ["config", "seed"])

        # Create dynamic wrapper function based on facade requirements
        if "budget" in required_params:

            def smac_train_function(config, seed: int = 0, budget: int = None) -> float:
                """Wrapper function for multi-fidelity facades (Hyperband, MultiFidelity)"""
                # Convert budget to int if it's provided as a float
                budget_int = int(budget) if budget is not None else budget
                result = train_fn(cfg=config, dataset=dataset, seed=int(seed), budget=budget_int)
                # Extract loss from tuple if train function returns (loss, model)
                if isinstance(result, tuple):
                    return result[0]  # Return only the loss for SMAC
                else:
                    return result  # Return the loss directly

        else:

            def smac_train_function(config, seed: int = 0) -> float:
                """Wrapper function for standard facades (BlackBox, HyperparameterOptimization)"""
                result = train_fn(cfg=config, dataset=dataset, seed=int(seed))
                # Extract loss from tuple if train function returns (loss, model)
                if isinstance(result, tuple):
                    return result[0]  # Return only the loss for SMAC
                else:
                    return result  # Return the loss directly

        smac = self.suggested_facade(
            scenario,
            smac_train_function,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )
        incumbent = smac.optimize()

        default_cost = smac.validate(self.config_space_obj.get_default_configuration())
        self.ui_agent.subheader("Default Cost")
        self.ui_agent.write(default_cost)
        self.experimenter_row_dict["default_train_accuracy"] = default_cost

        incubment_cost = smac.validate(incumbent)
        self.ui_agent.subheader("Incumbent Cost")
        self.ui_agent.write(incubment_cost)
        self.experimenter_row_dict["incumbent_train_accuracy"] = incubment_cost

        self.logger.log_response(
            f"Incumbent cost: {incubment_cost}",
            {"component": "scenario", "status": "success", "cost": incubment_cost},
        )
        self.logger.log_response(
            f"Default cost: {default_cost}",
            {"component": "scenario", "status": "success", "cost": default_cost},
        )
        self.logger.log_response(
            f"Incumbent config: {incumbent}",
            {"component": "scenario", "status": "success", "config": incumbent},
        )
        self.logger.log_response(
            f"Default config: {default_cost}",
            {"component": "scenario", "status": "success", "config": default_cost},
        )

        self.test_incumbent(train_fn, incumbent, self.dataset, self.dataset_test, self.config_space_obj)

    def test_incumbent(self, train_fn, incumbent: Any, dataset_train: Any, dataset_test: Any, cfg: Any):
        """Test the incumbent on the test set"""
        # Get the facade name to determine parameter requirements
        facade_name = self.suggested_facade.__name__
        required_params = FACADE_PARAMETER_REQUIREMENTS.get(facade_name, ["config", "seed"])

        # Call train function with appropriate parameters
        if "budget" in required_params:
            # For multi-fidelity facades, use maximum budget for final evaluation
            max_budget = getattr(self.scenario_obj, "max_budget", None)
            # Convert budget to int if it's provided as a float
            budget_int = int(max_budget) if max_budget is not None else max_budget
            result = train_fn(
                cfg=incumbent,
                dataset=dataset_train,
                seed=0,
                budget=budget_int,
                model_dir=self.logger.experiment_dir,
            )
        else:
            result = train_fn(
                cfg=incumbent,
                dataset=dataset_train,
                seed=0,
                model_dir=self.logger.experiment_dir,
            )

        # Handle both tuple and single value returns
        if isinstance(result, tuple):
            loss, model = result
        else:
            loss = result
            model = None

        # TODO
        if model is not None:
            predictions = model.predict(dataset_test["X"])
            if self.task_type == "classification":
                metrics = classification_report(dataset_test["y"], predictions)
            elif self.task_type == "regression":
                metrics = mean_squared_error(dataset_test["y"], predictions)
        else:
            metrics = None  # Cannot compute accuracy without model

        self.ui_agent.subheader("Metrics")
        self.ui_agent.write(f"Metrics: {metrics}")
        self.experimenter_row_dict["incumbent_config"] = str(incumbent)
        self.experimenter_row_dict["test_train_accuracy"] = metrics

        if metrics is not None:
            self.logger.log_response(
                f"Metrics: {metrics}",
                {
                    "component": "scenario",
                    "status": "success",
                    "metrics": f"{metrics}",
                },
            )
        else:
            self.logger.log_response(
                "Test accuracy: Unable to compute (no model returned)",
                {"component": "scenario", "status": "warning", "accuracy": None},
            )
        self.logger.log_response(
            f"Incumbent config: {incumbent}",
            {"component": "scenario", "status": "success", "config": incumbent},
        )
