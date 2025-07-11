from scripts.LLMClient import LLMClient
from typing import Any
from utils import (
    describe_dataset,
    save_code_to_file,
    format_dataset,
    extract_code_block,
    split_dataset_kfold,
    FACADE_PARAMETER_REQUIREMENTS,
    evaluate_model_predictions,
    get_default_metrics,
    execute_training_function,
    prepare_training_parameters,
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
from smac.facade.hyperband_facade import HyperbandFacade as HyperbandFacade
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as MultiFidelityFacade
from smac.facade.blackbox_facade import BlackBoxFacade
import numpy as np
import multiprocessing
import concurrent.futures
import threading


def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """
    Run a function with a timeout using concurrent.futures

    :param func: Function to run
    :param timeout_seconds: Timeout in seconds
    :param args: Function arguments
    :param kwargs: Function keyword arguments
    :return: Function result
    :raises TimeoutError: If function exceeds timeout
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


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
        n_folds: int = 5,
        fold: int = 0,
        time_budget: int = 3600,  # Default 1 hour timeout
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
        self.dataset_description = describe_dataset(self.dataset, dataset_type, task_type)
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

    def _test_train_function_comprehensive(self, cfg: Any, train_fn: Any) -> dict:
        """
        Comprehensively test the train function with multiple configurations to ensure robustness.

        :param cfg: Configuration space object
        :param train_fn: Train function to test
        :return: Dictionary with testing results
        """
        test_results = {
            "success": False,
            "tests_run": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "best_loss": float("inf"),
            "error": None,
            "test_details": [],
        }

        # Get dataset for train function
        dataset_for_train = self._get_dataset_for_train_function(self.dataset, self.dataset_test, include_test_data=True)

        # Determine if function expects budget parameter
        facade_name = getattr(self, "suggested_facade", None)
        if facade_name:
            facade_name = facade_name.__name__
            required_params = FACADE_PARAMETER_REQUIREMENTS.get(facade_name, ["config", "seed"])
            expects_budget = "budget" in required_params
        else:
            expects_budget = True  # Test both cases if unsure

        # Test configurations to try
        test_configs = []

        # 1. Default configuration
        try:
            default_config = cfg.get_default_configuration()
            test_configs.append(("default", default_config))
        except Exception as e:
            self.logger.log_error(
                f"Error getting default configuration: {e}",
                error_type="TRAIN_FUNCTION_TESTING_ERROR",
            )
            pass

        # 2. Multiple random configurations
        for i in range(5):
            try:
                random_config = cfg.sample_configuration()
                test_configs.append((f"random_{i+1}", random_config))
            except Exception as e:
                self.logger.log_error(
                    f"Error getting random configuration: {e}",
                    error_type="TRAIN_FUNCTION_TESTING_ERROR",
                )
                pass

        # Budget values to test (if applicable)
        budget_values = [None, 1, 10, 50] if expects_budget else [None]

        last_error = None

        for config_name, config in test_configs:
            test_results["tests_run"] += 1

            for budget in budget_values:
                try:
                    # Add timeout protection for individual tests
                    def test_train_call():
                        if budget is not None:
                            # Test with budget parameter
                            try:
                                return train_fn(
                                    cfg=config,
                                    dataset=dataset_for_train,
                                    seed=0,
                                    budget=budget,
                                )
                            except TypeError:
                                # Function doesn't accept budget, try without
                                return train_fn(cfg=config, dataset=dataset_for_train, seed=0)
                        else:
                            # Test without budget parameter
                            try:
                                return train_fn(cfg=config, dataset=dataset_for_train, seed=0)
                            except TypeError:
                                # Maybe it requires budget, try with None
                                return train_fn(
                                    cfg=config,
                                    dataset=dataset_for_train,
                                    seed=0,
                                    budget=None,
                                )

                    result = run_with_timeout(test_train_call, 30)  # 30 second timeout per test

                    # Handle different return formats
                    if isinstance(result, dict) and "loss" in result:
                        loss = float(result["loss"])
                    elif isinstance(result, (int, float)):
                        loss = float(result)  # type: ignore
                    elif hasattr(result, "__float__"):
                        loss = float(result)  # type: ignore
                    else:
                        # Fallback for unexpected return types
                        loss = float("inf")

                    # Track best loss
                    if loss < test_results["best_loss"]:
                        test_results["best_loss"] = loss

                    test_results["successful_tests"] += 1
                    test_results["test_details"].append(
                        {
                            "config_name": config_name,
                            "budget": budget,
                            "loss": loss,
                            "status": "success",
                        }
                    )

                    # If we have at least one successful test, we can proceed
                    if test_results["successful_tests"] >= 1:
                        test_results["success"] = True

                    # Don't test all budget values if one works
                    break

                except Exception as e:
                    last_error = str(e)
                    test_results["failed_tests"] += 1
                    test_results["test_details"].append(
                        {
                            "config_name": config_name,
                            "budget": budget,
                            "error": str(e),
                            "status": "failed",
                        }
                    )

                    # Continue to next budget value
                    continue

        # Final success check
        if test_results["successful_tests"] == 0:
            test_results["success"] = False
            test_results["error"] = last_error or "All train function tests failed"

        # Log detailed test results
        self.logger.log_response(
            f"Train function testing completed: {test_results['successful_tests']}/{test_results['tests_run']} tests passed",
            {"component": "train_function_testing", "test_results": test_results},
        )

        return test_results

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
        self.train_fn = self.namespace["train"]

        self.ui_agent.success("AutoML Agent setup complete!")
        self.ui_agent.subheader("Loss Value")
        self.ui_agent.write(self.last_loss)

        self.ui_agent.subheader("Starting Optimization Process")
        self.ui_agent.write("Starting Optimization Process")

        loss, metrics = self.run_scenario(self.scenario_obj, self.train_fn, self.dataset, self.config_space_obj)

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

                    # Comprehensive testing with multiple configurations
                    self.ui_agent.info("Running comprehensive train function tests with multiple configurations...")
                    test_results = self._test_train_function_comprehensive(cfg, train_fn)

                    if not test_results["success"]:
                        # If comprehensive testing failed, raise the last error encountered
                        raise Exception(f"Train function testing failed: {test_results['error']}")

                    # Store the best result from testing
                    self.last_loss = test_results["best_loss"]
                    self.ui_agent.success(
                        f"✅ Train function testing successful! {test_results['successful_tests']}/{test_results['tests_run']} configurations passed"
                    )

                    self.last_model = None
                    self.logger.log_response(
                        f"Training function tested successfully with {test_results['successful_tests']} configurations. Best loss: {self.last_loss}",
                        {
                            "component": component,
                            "status": "success",
                            "loss": self.last_loss,
                            "tests_run": test_results["tests_run"],
                            "successful_tests": test_results["successful_tests"],
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
                    def train_call():
                        return train_fn(
                            cfg=config,
                            dataset=dataset_for_train,
                            seed=int(seed),
                            budget=budget_int,
                        )

                    timeout_seconds = min(300, self.time_budget // 10)  # Max 5 minutes per trial or 1/10 of total budget
                    result = run_with_timeout(train_call, timeout_seconds)

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
                    def train_call():
                        return train_fn(cfg=config, dataset=dataset_for_train, seed=int(seed))

                    timeout_seconds = min(300, self.time_budget // 10)  # Max 5 minutes per trial or 1/10 of total budget
                    result = run_with_timeout(train_call, timeout_seconds)

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

            def optimization_call():
                return smac.optimize()

            incumbent = run_with_timeout(optimization_call, optimization_timeout)

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
                metrics = get_default_metrics(self.task_type)
                metrics["loss"] = loss
                self._log_evaluation_success(metrics, incumbent)

        except Exception as e:
            # Handle training function execution errors
            loss = float("inf")
            metrics = get_default_metrics(self.task_type)
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
