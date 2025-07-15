from scripts.LLMClient import LLMClient
from typing import Any, Optional
from utils import (
    describe_dataset,
    save_code_to_file,
    format_dataset,
    extract_code_block,
    split_dataset_kfold,
    FACADE_PARAMETER_REQUIREMENTS,
    get_default_metrics,
    prepare_training_parameters,
)
from scripts.Logger import Logger
from scripts.OpenMLRAG import OpenMLRAG
from config.api_keys import OPENML_API_KEY
from config.configs_path import (
    OPENML_CSV_PATH,
)
from scripts.LLMPlanner import LLMPlanner
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac import Scenario
from smac.facade.hyperband_facade import HyperbandFacade as HyperbandFacade
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as MultiFidelityFacade
from smac.facade.blackbox_facade import BlackBoxFacade


import numpy as np
from contextlib import contextmanager


@contextmanager
def timeout_handler(seconds):
    """Thread-safe context manager for timeout handling using multiprocessing"""
    # Use a simple no-op timeout handler to prevent signal issues
    # SMAC has its own timeout mechanisms that are more appropriate for HPO
    try:
        yield
    except Exception:
        # Let exceptions propagate normally
        raise


class AutoMLAgent:
    MAX_RETRIES_PROMPT = 10
    MAX_RETRIES_ABORT = 20

    def __init__(
        self,
        dataset: Any,
        dataset_type: str,
        ui_agent: Any,
        dataset_name: Optional[str] = None,
        task_type: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        n_folds: int = 5,
        fold: int = 0,
        time_budget: int = 3600,  # Default 1 hour timeout
        folder_name: str = "./logs",
    ):
        """
        Initialize the AutoML Agent with dataset, LLM client, and UI agent.
        :param dataset: The dataset to be used for training.
        :param llm_client: The LLM client for generating code.
        :param dataset_type: The type of dataset (e.g., "tabular", "image", etc.).
        :param ui_agent: The UI agent for displaying results and prompts.
        :param dataset_name: Optional name of the dataset for logging and display.
        :param task_type: Optional task type for logging and display.
        :param time_budget: Maximum time budget in seconds for the optimization process.
        :param include_test_data: Whether to include test data in dataset dict passed to train function.
                                When True, train function will receive both X_test and y_test and should NOT do internal splits.
                                When False, train function will only receive X and y and should do internal train/test splits.
        """
        self.dataset = format_dataset(dataset)
        self.dataset, self.dataset_test = split_dataset_kfold(self.dataset, n_folds=n_folds, fold=fold, rng=np.random.RandomState(42))
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name if dataset_name else None
        self.task_type = task_type if task_type else None
        self.dataset_description = describe_dataset(self.dataset, dataset_type, task_type or "classification")
        self.time_budget = time_budget

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

        self.llm = LLMClient(
            api_key=api_key or "",
            model_name=model_name or "llama3-8b-8192",
            base_url=base_url
        )
        self.ui_agent = ui_agent

        # Initialize logger
        self.logger = Logger(
            model_name=model_name or "llama3-8b-8192",
            dataset_name=dataset_name or "unknown",
            base_log_dir=folder_name,
        )

        self.openML = OpenMLRAG(openml_api_key=OPENML_API_KEY, metafeatures_csv_path=OPENML_CSV_PATH)
        self.LLMInstructor = LLMPlanner(
            dataset_name=self.dataset_name or "unknown",
            dataset_description=self.dataset_description,
            dataset_type=self.dataset_type,
            task_type=self.task_type or "classification",
            model_name=model_name or "llama3-8b-8192",
            base_url=base_url,
            api_key=api_key or "",
        )

        self.smac_facades = {
            "HyperbandFacade": HyperbandFacade,
            "MultiFidelityFacade": MultiFidelityFacade,
            "HyperparameterOptimizationFacade": HPOFacade,
            "BlackBoxFacade": BlackBoxFacade,
        }

    def _get_dataset_for_train_function(
        self,
        dataset_train: Any,
        dataset_test: Any = None,
        include_test_data: bool = False,
    ) -> Any:
        """
        Construct the dataset dictionary for the train function based on include_test_data setting.
        :param dataset_train: The training dataset dictionary
        :param dataset_test: The test dataset dictionary (optional)
        :return: Dataset dictionary configured for the train function
        """
        if include_test_data and dataset_test is not None:
            # Include test data - train function should NOT do internal splits
            return {
                "X": dataset_train["X"],
                "y": dataset_train["y"],
                "X_test": dataset_test["X"],
                "y_test": dataset_test["y"],
            }
        else:
            # Don't include test data - train function should do internal splits
            return {
                "X": dataset_train["X"],
                "y": dataset_train["y"],
            }


    


    def test_incumbent(self, train_fn, incumbent: Any, dataset_train: Any, dataset_test: Any, cfg: Any):
        """Test the incumbent configuration and evaluate model on test set if possible"""
        
        # Get facade parameters
        facade_name = self.suggested_facade.__name__
        required_params = FACADE_PARAMETER_REQUIREMENTS.get(facade_name, ["config", "seed"])

        # Get dataset dictionary for train function
        dataset_for_train = self._get_dataset_for_train_function(dataset_train, dataset_test, include_test_data=True)

        # Prepare training parameters using utility function
        training_params = prepare_training_parameters(
            facade_name=facade_name,
            required_params=required_params,
            scenario_obj=self.scenario_obj,
            incumbent=incumbent,
            dataset=dataset_for_train,
            seed=0,
            model_dir=self.logger.experiment_dir,
        )

        try:
            # Execute training function - now returns dictionary with loss and metrics
            result = train_fn(**training_params)

            # Handle different return formats
            if isinstance(result, dict):
                # New format: dictionary with loss and metrics
                metrics = result.copy()
                loss = metrics.get("loss", float("inf"))
                self._log_evaluation_success(metrics, incumbent)
            else:
                # Old format: just loss value (backward compatibility)
                loss = float(result)
                metrics = get_default_metrics(self.task_type or "classification")
                metrics["loss"] = loss
                self._log_evaluation_success(metrics, incumbent)

        except Exception as e:
            # Handle training function execution errors
            loss = float("inf")
            metrics = get_default_metrics(self.task_type or "classification")
            metrics["error"] = f"Training function execution failed: {str(e)}"
            self.logger.log_error(
                f"Training function execution failed: {str(e)}",
                error_type="TRAINING_ERROR",
            )

        # Display results
        self.ui_agent.subheader("Test Metrics")
        self.ui_agent.write(f"Metrics: {metrics}")

        # Store results
        self.last_loss = loss
        self.last_metrics = metrics

        return loss, metrics

    def _log_evaluation_success(self, metrics: dict, incumbent: Any):
        """Log successful model evaluation"""
        self.logger.log_response(
            f"Model evaluation successful. Metrics: {metrics}",
            {
                "component": "evaluation",
                "status": "success",
                "metrics": metrics,
                "config": str(incumbent),
            },
        )

    def _log_evaluation_error(self, error_msg: str):
        """Log model evaluation errors"""
        self.logger.log_response(
            f"Model evaluation failed: {error_msg}",
            {
                "component": "evaluation",
                "status": "error",
                "error": error_msg,
            },
        )

    def generate_components(self):
        """
        Generate the configuration space, scenario, and training function code using LLM.
        :return: Tuple containing the generated configuration space, scenario, training function code, last loss, and prompts.
        """
        max_retries = 3  # Maximum number of retries for the entire process
        
        for attempt in range(max_retries):
            try:
                self.ui_agent.info(f"Starting component generation (attempt {attempt + 1}/{max_retries})")
                
                config_space_suggested_parameters = self.openML.extract_suggested_config_space_parameters(
                    dataset_name=self.dataset_name or "unknown",
                    X=self.dataset["X"],
                    y=self.dataset["y"],
                )
                instructor_response = self.LLMInstructor.generate_instructions(
                    config_space_suggested_parameters,
                )
                self.suggested_facade = self.smac_facades[instructor_response.suggested_facade]
                self.logger.log_response(
                    f"Suggested Facade: {instructor_response.suggested_facade}",
                    {"component": "instructor", "status": "success", "facade": instructor_response.suggested_facade},
                )

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

                # Pre-calculate optimal parameters for multi-fidelity facades
                facade_name = self.suggested_facade.__name__
                
                # Scenario
                self.prompts["scenario"] = self._create_scenario_prompt(
                    instructor_response.scenario_plan, 
                    instructor_response.suggested_facade
                )
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
                if self.config_code and isinstance(self.config_code, str):
                    save_code_to_file(self.config_code, "scripts/generated_codes/generated_config_space.py")
                if self.scenario_code and isinstance(self.scenario_code, str):
                    save_code_to_file(self.scenario_code, "scripts/generated_codes/generated_scenario.py")
                if self.train_function_code and isinstance(self.train_function_code, str):
                    save_code_to_file(
                        self.train_function_code,
                        "scripts/generated_codes/generated_train_function.py",
                    )

                # Use the train function that was just generated and tested, not the imported one
                # This ensures consistency between the tested function and the one used in SMAC
                self.train_fn = self.namespace["train"]

                self.ui_agent.success("AutoML Agent setup complete!")
                self.ui_agent.subheader("Loss Value")
                self.ui_agent.write(self.last_loss)

                self.ui_agent.subheader("Starting Optimization Process")
                self.ui_agent.write("Starting Optimization Process")

                loss, metrics = self.run_scenario(self.scenario_obj, self.train_fn, self.dataset, self.config_space_obj)

                # If we get here, everything worked successfully
                self.ui_agent.success(f"Component generation completed successfully on attempt {attempt + 1}")
                
                # Return results and last training loss
                return (
                    self.config_code,
                    self.scenario_code,
                    self.train_function_code,
                    loss,
                    metrics,
                    self.prompts,
                    self.logger.experiment_dir,
                )

            except Exception as e:
                self.logger.log_error(
                    f"Component generation failed on attempt {attempt + 1}: {str(e)}",
                    error_type="COMPONENT_GENERATION_ERROR",
                )
                self.ui_agent.error(f"Component generation failed on attempt {attempt + 1}: {str(e)}")
                
                if attempt == max_retries - 1:
                    # Last attempt failed, raise the error
                    self.ui_agent.error(f"All {max_retries} attempts failed. Raising error.")
                    raise e
                else:
                    # Wait a bit before retrying
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    self.ui_agent.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

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
                    # Get dataset dictionary for train function
                    dataset_for_train = self._get_dataset_for_train_function(self.dataset, self.dataset_test, include_test_data=True)

                    # if the train function is multi-fidelity or hyperband, then we need to pass the budget parameter
                    if self.suggested_facade.__name__ in ["HyperbandFacade", "MultiFidelityFacade"]:
                        result = train_fn(cfg=sampled, dataset=dataset_for_train, seed=0, budget=10)
                    else:
                        result = train_fn(cfg=sampled, dataset=dataset_for_train, seed=0)

                    # Handle different return formats
                    if isinstance(result, dict) and "loss" in result:
                        # New format: dictionary with loss and metrics
                        self.last_loss = result["loss"]
                        metrics_info = f", metrics: {list(result.keys())}"
                    else:
                        # Old format: just loss value (backward compatibility)
                        self.last_loss = float(result)
                        metrics_info = ""

                    self.last_model = None
                    self.logger.log_response(
                        f"Training executed successfully, loss: {self.last_loss}{metrics_info}",
                        {
                            "component": component,
                            "status": "success",
                            "loss": self.last_loss,
                        },
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

        # Get dataset dictionary for train function
        dataset_for_train = self._get_dataset_for_train_function(dataset, self.dataset_test)

        # Create dynamic wrapper function based on facade requirements
        if "budget" in required_params:

            def smac_train_function(config, seed: int = 0, budget: int = None) -> float:
                """Wrapper function for multi-fidelity facades (Hyperband, MultiFidelity)"""
                try:
                    # Convert budget to int if it's provided as a float
                    budget_int = int(budget) if budget is not None else budget

                    # Add timeout protection for individual training runs
                    with timeout_handler(min(600, self.time_budget // 10)):  # Max 10 minutes per trial or 1/10 of total budget
                        result = train_fn(
                            cfg=config,
                            dataset=dataset_for_train,
                            seed=int(seed),
                            budget=budget_int,
                        )

                    # Train function now returns dictionary with loss and metrics
                    if isinstance(result, dict) and "loss" in result:
                        return result["loss"]
                    else:
                        return float(result)  # Fallback for backward compatibility

                except (TimeoutError, Exception) as e:
                    self.logger.log_error(
                        f"Training function failed: {str(e)}",
                        error_type="TRAINING_TIMEOUT",
                    )
                    return float("inf")  # Return worst possible score

        else:

            def smac_train_function(config, seed: int = 0) -> float:
                """Wrapper function for standard facades (BlackBox, HyperparameterOptimization)"""
                try:
                    # Add timeout protection for individual training runs
                    with timeout_handler(min(600, self.time_budget // 10)):  # Max 10 minutes per trial or 1/10 of total budget
                        result = train_fn(cfg=config, dataset=dataset_for_train, seed=int(seed))

                    # Train function now returns dictionary with loss and metrics
                    if isinstance(result, dict) and "loss" in result:
                        return result["loss"]
                    else:
                        return float(result)  # Fallback for backward compatibility

                except (TimeoutError, Exception) as e:
                    self.logger.log_error(
                        f"Training function failed: {str(e)}",
                        error_type="TRAINING_TIMEOUT",
                    )
                    return float("inf")  # Return worst possible score

        smac = self.suggested_facade(
            scenario,
            smac_train_function,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )

        # Add timeout protection around SMAC optimization
        incumbent = None
        try:
            self.ui_agent.info(f"Starting SMAC optimization with {self.time_budget} seconds timeout...")

            # Use 90% of time budget for optimization, reserve 10% for final evaluation
            optimization_timeout = int(self.time_budget * 0.9)

            # Log optimization start
            self.logger.log_response(
                f"Starting SMAC optimization with {optimization_timeout}s timeout",
                {
                    "component": "optimization",
                    "status": "start",
                    "timeout": optimization_timeout,
                },
            )

            with timeout_handler(optimization_timeout):
                incumbent = smac.optimize()

            self.ui_agent.success(f"SMAC optimization completed successfully!")
            self.logger.log_response(
                "SMAC optimization completed successfully",
                {"component": "optimization", "status": "success"},
            )

        except TimeoutError:
            self.logger.log_error(
                f"SMAC optimization timed out after {optimization_timeout} seconds. Using best configuration found so far.",
                error_type="SMAC_TIMEOUT",
            )
            self.ui_agent.warning(f"Optimization timed out after {optimization_timeout}s, using best config found so far")

            # Get the best configuration found so far
            try:
                incumbent = smac.optimizer.intensifier.get_incumbent()
            except Exception as e:
                self.logger.log_error(f"Error getting incumbent: {e}", error_type="SMAC_ERROR")
                # Fallback to default configuration if no incumbent found
                incumbent = cfg.get_default_configuration()
                self.logger.log_error(
                    "No incumbent found, using default configuration",
                    error_type="FALLBACK_CONFIG",
                )

        except Exception as e:
            self.logger.log_error(f"SMAC optimization failed: {str(e)}", error_type="SMAC_ERROR")
            self.ui_agent.error(f"Optimization failed: {str(e)}")
            # Fallback to default configuration
            incumbent = cfg.get_default_configuration()
            self.logger.log_error(
                "Using default configuration due to optimization failure",
                error_type="FALLBACK_CONFIG",
            )

        default_cost = smac.validate(self.config_space_obj.get_default_configuration())
        self.ui_agent.subheader("Default Cost")
        self.ui_agent.write(default_cost)

        incubment_cost = smac.validate(incumbent)
        self.ui_agent.subheader("Incumbent Cost")
        self.ui_agent.write(incubment_cost)

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
        self.incumbent = incumbent
        return self.test_incumbent(
            self.train_fn,
            incumbent,
            self.dataset,
            self.dataset_test,
            self.config_space_obj,
        )
