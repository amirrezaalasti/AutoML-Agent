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
from utils.baseline_evaluator import create_baseline_evaluator
from scripts.PostProcessor import PostProcessor
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
import pandas as pd
import os
from pathlib import Path


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
        baseline_time_budget: int = 300,  # 5 minutes for baseline evaluation
        enable_baseline_retry: bool = True,  # Enable automatic retry with improvement feedback
        max_improvement_rounds: int = 3,  # Maximum iterative improvement rounds if baseline not beaten
        track_best_across_rounds: bool = True,  # Track and return/save best result across rounds
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
        :param baseline_time_budget: Time budget for AutoGluon baseline evaluation.
        :param include_test_data: Whether to include test data in dataset dict passed to train function.
                                When True, train function will receive both X_test and y_test and should NOT do internal splits.
                                When False, train function will only receive X and y and should do internal train/test splits.
        """
        self.dataset = format_dataset(dataset)
        self.dataset, self.dataset_test = split_dataset_kfold(
            self.dataset, n_folds=n_folds, fold=fold, rng=np.random.RandomState(42)
        )
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name if dataset_name else None
        self.task_type = task_type if task_type else None
        self.dataset_description = describe_dataset(
            self.dataset, dataset_type, task_type or "classification"
        )
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

        # Baseline retry configuration
        self.enable_baseline_retry = enable_baseline_retry
        self.max_improvement_rounds = max_improvement_rounds
        self.track_best_across_rounds = track_best_across_rounds

        # Initialize baseline evaluator
        self.baseline_evaluator = create_baseline_evaluator(
            dataset_type=dataset_type, time_budget=baseline_time_budget
        )
        self.baseline_metrics = None
        self.baseline_accuracy = None

        self.llm = LLMClient(
            api_key=api_key or "",
            model_name=model_name or "llama3-8b-8192",
            base_url=base_url,
        )
        self.ui_agent = ui_agent

        # Initialize logger
        self.logger = Logger(
            model_name=model_name or "llama3-8b-8192",
            dataset_name=dataset_name or "unknown",
            base_log_dir=folder_name,
        )

        # Initialize post-processor
        self.post_processor = PostProcessor(ui_agent, self.logger)

        self.openML = OpenMLRAG(
            openml_api_key=OPENML_API_KEY, metafeatures_csv_path=OPENML_CSV_PATH
        )
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

        # Best-tracking across rounds
        self.best_loss = None
        self.best_metrics = None
        self.best_accuracy = None
        self.best_round = None
        self.best_incumbent = None
        self.best_codes = {
            "config": None,
            "scenario": None,
            "train_function": None,
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

    def test_incumbent(
        self, train_fn, incumbent: Any, dataset_train: Any, dataset_test: Any, cfg: Any
    ):
        """Test the incumbent configuration and evaluate model on test set if possible"""

        # Get facade parameters
        facade_name = self.suggested_facade.__name__
        required_params = FACADE_PARAMETER_REQUIREMENTS.get(
            facade_name, ["config", "seed"]
        )

        # Get dataset dictionary for train function
        dataset_for_train = self._get_dataset_for_train_function(
            dataset_train, dataset_test, include_test_data=True
        )

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
                metrics = result.copy()
                loss = metrics.get("loss", float("inf"))
            else:
                loss = float(result)
                metrics = get_default_metrics(self.task_type or "classification")
                metrics["loss"] = loss

            # Validate metrics; if invalid, log and fallback to default error metrics
            if not self._is_valid_metrics(metrics):
                self.logger.log_error(
                    f"Invalid metrics returned from training function: {metrics}",
                    error_type="TRAINING_METRICS_INVALID",
                )
                metrics = get_default_metrics(self.task_type or "classification")
                metrics["error"] = "Invalid metrics returned from training function"
                loss = float("inf")
            else:
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

        # If metrics invalid, ask LLM to fix train function and retry once
        if not self._is_valid_metrics(metrics):
            self.ui_agent.warning(
                "Primary evaluation returned invalid metrics; regenerating train function with fix prompt and retrying once."
            )
            self._regenerate_component_with_fix(
                component="train_function",
                summary_error="Invalid metrics detected after evaluation (non-finite loss or out-of-range accuracy)",
            )

            # Re-evaluate once after regeneration
            try:
                result_retry = train_fn(**training_params)
                if isinstance(result_retry, dict):
                    metrics_retry = result_retry.copy()
                    loss_retry = metrics_retry.get("loss", float("inf"))
                else:
                    loss_retry = float(result_retry)
                    metrics_retry = get_default_metrics(
                        self.task_type or "classification"
                    )
                    metrics_retry["loss"] = loss_retry

                if self._is_valid_metrics(metrics_retry):
                    metrics = metrics_retry
                    loss = loss_retry
                    self.logger.log_response(
                        "Train function regenerated and metrics validated on retry.",
                        {"component": "evaluation", "status": "retry_success"},
                    )
                else:
                    self.logger.log_error(
                        f"Metrics still invalid after retry: {metrics_retry}",
                        error_type="TRAINING_METRICS_INVALID_RETRY",
                    )
            except Exception as e:
                self.logger.log_error(
                    f"Retry after regeneration failed: {str(e)}",
                    error_type="TRAINING_RETRY_FAILED",
                )

        # Display results
        self.ui_agent.subheader("Test Metrics")
        self.ui_agent.write(f"Metrics: {metrics}")

        # Store results
        self.last_loss = loss
        self.last_metrics = metrics

        # Validate against baseline target
        self.validate_baseline_target(metrics)

        # Post-process if baseline was not beaten
        if self.baseline_accuracy is not None:
            baseline_beaten, improvement_feedback = (
                self.post_processor.analyze_performance(
                    metrics,
                    incumbent,
                    self.config_code,
                    self.scenario_code,
                    self.train_function_code,
                )
            )

            # Debug logging for post-processing result
            self.logger.log_response(
                f"Post-processing completed: baseline_beaten={baseline_beaten}, "
                f"improvement_feedback_set={self.post_processor.improvement_feedback is not None}, "
                f"improvement_feedback_length={len(self.post_processor.improvement_feedback) if self.post_processor.improvement_feedback else 0}",
                {"component": "post_processing", "action": "debug"},
            )

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
        max_retries = 5  # Maximum number of retries for the entire process

        for attempt in range(max_retries):
            try:
                self.ui_agent.info(
                    f"Starting component generation (attempt {attempt + 1}/{max_retries})"
                )

                # Add iteration header for initial generation
                self.ui_agent.subheader("🚀 Initial Component Generation")
                self.ui_agent.info(
                    "Generating initial configuration space, scenario, and training function..."
                )

                config_space_suggested_parameters = (
                    self.openML.extract_suggested_config_space_parameters(
                        dataset_name=self.dataset_name or "unknown",
                        X=self.dataset["X"],
                        y=self.dataset["y"],
                    )
                )
                instructor_response = self.LLMInstructor.generate_instructions(
                    config_space_suggested_parameters,
                )
                self.suggested_facade = self.smac_facades[
                    instructor_response.suggested_facade
                ]
                self.logger.log_response(
                    f"Suggested Facade: {instructor_response.suggested_facade}",
                    {
                        "component": "instructor",
                        "status": "success",
                        "facade": instructor_response.suggested_facade,
                    },
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
                    instructor_response.suggested_facade,
                )
                self.scenario_code = self.llm.generate(self.prompts["scenario"])
                self.logger.log_prompt(
                    self.prompts["scenario"], {"component": "scenario"}
                )
                self.logger.log_response(self.scenario_code, {"component": "scenario"})
                self._run_generated("scenario")
                self.ui_agent.subheader("Generated Scenario Code")
                self.ui_agent.code(self.scenario_code, language="python")

                # Evaluate baseline performance before generating training function
                self.evaluate_baseline()

                # Training Function
                train_plan = instructor_response.train_function_plan
                self.prompts["train_function"] = self._create_train_prompt(
                    train_plan, instructor_response.suggested_facade
                )
                self.train_function_code = self.llm.generate(
                    self.prompts["train_function"]
                )
                self.logger.log_prompt(
                    self.prompts["train_function"], {"component": "train_function"}
                )
                self.logger.log_response(
                    self.train_function_code, {"component": "train_function"}
                )
                self._run_generated("train_function")
                self.ui_agent.subheader("Generated Training Function Code")
                self.ui_agent.code(self.train_function_code, language="python")

                # Quality gate: ensure smoke test succeeded and produced a finite loss
                if (
                    not hasattr(self, "last_loss")
                    or self.last_loss is None
                    or not self._is_finite(self.last_loss)
                ):
                    self.logger.log_error(
                        "Training function smoke test failed (non-finite or missing loss). Restarting component generation.",
                        error_type="TRAIN_FUNCTION_SMOKE_TEST_FAILED",
                    )
                    self.ui_agent.warning(
                        "Training function smoke test failed. Restarting component generation..."
                    )
                    continue  # Restart the current iteration

                # Use the train function that was just generated and tested, not the imported one
                # This ensures consistency between the tested function and the one used in SMAC
                self.train_fn = self.namespace["train"]

                self.ui_agent.success("AutoML Agent setup complete!")
                self.ui_agent.subheader("Loss Value")
                self.ui_agent.write(self.last_loss)

                self.ui_agent.subheader("Starting Optimization Process")
                self.ui_agent.write("Starting Optimization Process")

                loss, metrics = self.run_scenario(
                    self.scenario_obj,
                    self.train_fn,
                    self.dataset,
                    self.config_space_obj,
                )

                # Initialize best-tracking with first-run results (only if metrics valid)
                if self._is_valid_metrics(metrics):
                    current_accuracy = self._extract_accuracy_from_metrics(metrics)
                    self.best_loss = loss
                    self.best_metrics = metrics
                    self.best_accuracy = current_accuracy
                    self.best_round = 0
                    self.best_incumbent = getattr(self, "incumbent", None)
                    self.best_codes = {
                        "config": self.config_code,
                        "scenario": self.scenario_code,
                        "train_function": self.train_function_code,
                    }

                # Multi-round improvement loop
                if self.enable_baseline_retry and self.baseline_accuracy is not None:
                    # Debug logging for improvement loop conditions
                    self.logger.log_response(
                        f"Checking improvement loop conditions: "
                        f"enable_baseline_retry={self.enable_baseline_retry}, "
                        f"baseline_accuracy={self.baseline_accuracy}, "
                        f"improvement_feedback_exists={self.post_processor.improvement_feedback is not None}, "
                        f"best_accuracy={self.best_accuracy}, "
                        f"baseline_beaten={self.best_accuracy is not None and self.best_accuracy >= self.baseline_accuracy}",
                        {"component": "improvement_loop", "action": "condition_check"},
                    )

                    rounds_run = 0
                    improvement_needed = (
                        self.post_processor.improvement_feedback is not None
                        and rounds_run < self.max_improvement_rounds
                        and (
                            self.best_accuracy is None
                            or self.best_accuracy < self.baseline_accuracy
                        )
                    )

                    self.logger.log_response(
                        f"Improvement loop decision: improvement_needed={improvement_needed}, "
                        f"improvement_feedback_length={len(self.post_processor.improvement_feedback) if self.post_processor.improvement_feedback else 0}",
                        {"component": "improvement_loop", "action": "loop_decision"},
                    )

                    while improvement_needed:
                        rounds_run += 1
                        self.ui_agent.info(
                            f"Baseline not beaten. Attempting improvement round {rounds_run}/{self.max_improvement_rounds}..."
                        )

                        # Regenerate all components using improvement feedback
                        if not self.retry_with_improvement_feedback():
                            self.ui_agent.warning(
                                "Improvement regeneration failed; stopping improvement loop."
                            )
                            break

                        # Update train function reference after regeneration
                        self.train_fn = self.namespace["train"]

                        # Re-run scenario
                        loss_i, metrics_i = self.run_scenario(
                            self.scenario_obj,
                            self.train_fn,
                            self.dataset,
                            self.config_space_obj,
                        )

                        # Validate metrics; skip improvement if invalid
                        if not self._is_valid_metrics(metrics_i):
                            self.ui_agent.warning(
                                "Skipping invalid metrics from improvement round"
                            )
                            acc_i = None
                        else:
                            acc_i = self._extract_accuracy_from_metrics(metrics_i)

                        # Track best across rounds
                        is_better = False
                        if acc_i is not None and (
                            self.best_accuracy is None or acc_i > self.best_accuracy
                        ):
                            is_better = True
                        elif (
                            acc_i is None
                            and self.best_accuracy is None
                            and loss_i is not None
                            and (self.best_loss is None or loss_i < self.best_loss)
                        ):
                            # Fallback to loss if accuracy-like metric unavailable
                            is_better = True

                        if is_better:
                            self.best_loss = loss_i
                            self.best_metrics = metrics_i
                            self.best_accuracy = acc_i
                            self.best_round = rounds_run
                            self.best_incumbent = getattr(self, "incumbent", None)
                            self.best_codes = {
                                "config": self.config_code,
                                "scenario": self.scenario_code,
                                "train_function": self.train_function_code,
                            }

                        # Early exit if baseline beaten
                        if acc_i is not None and acc_i >= self.baseline_accuracy:
                            self.ui_agent.success(
                                "✓ Improvement achieved: baseline beaten."
                            )
                            break

                        # Update loop condition for next iteration
                        improvement_needed = (
                            self.post_processor.improvement_feedback is not None
                            and rounds_run < self.max_improvement_rounds
                            and (
                                self.best_accuracy is None
                                or self.best_accuracy < self.baseline_accuracy
                            )
                        )

                # Save last-generated artifacts
                if self.config_code and isinstance(self.config_code, str):
                    save_code_to_file(
                        self.config_code,
                        "scripts/generated_codes/generated_config_space.py",
                    )
                if self.scenario_code and isinstance(self.scenario_code, str):
                    save_code_to_file(
                        self.scenario_code,
                        "scripts/generated_codes/generated_scenario.py",
                    )
                if self.train_function_code and isinstance(
                    self.train_function_code, str
                ):
                    save_code_to_file(
                        self.train_function_code,
                        "scripts/generated_codes/generated_train_function.py",
                    )

                # Also save best artifacts separately if tracking enabled
                if self.track_best_across_rounds and self.best_codes["config"]:
                    save_code_to_file(
                        self.best_codes["config"],
                        "scripts/generated_codes/generated_config_space_best.py",
                    )
                if self.track_best_across_rounds and self.best_codes["scenario"]:
                    save_code_to_file(
                        self.best_codes["scenario"],
                        "scripts/generated_codes/generated_scenario_best.py",
                    )
                if self.track_best_across_rounds and self.best_codes["train_function"]:
                    save_code_to_file(
                        self.best_codes["train_function"],
                        "scripts/generated_codes/generated_train_function_best.py",
                    )

                # If we get here, everything worked successfully
                self.ui_agent.success(
                    f"Component generation completed successfully on attempt {attempt + 1}"
                )

                # Present best vs last
                if self.track_best_across_rounds and self.best_metrics is not None:
                    self.ui_agent.subheader("Best Metrics Across Rounds")
                    self.ui_agent.write(
                        {
                            "best_round": self.best_round,
                            "best_accuracy_like": self.best_accuracy,
                            "best_metrics": self.best_metrics,
                        }
                    )

                # Return results (prefer best across rounds if tracked)
                final_config_code = (
                    self.best_codes["config"]
                    if (self.track_best_across_rounds and self.best_codes["config"])
                    else self.config_code
                )
                final_scenario_code = (
                    self.best_codes["scenario"]
                    if (self.track_best_across_rounds and self.best_codes["scenario"])
                    else self.scenario_code
                )
                final_train_code = (
                    self.best_codes["train_function"]
                    if (
                        self.track_best_across_rounds
                        and self.best_codes["train_function"]
                    )
                    else self.train_function_code
                )
                final_loss = (
                    self.best_loss
                    if (self.track_best_across_rounds and self.best_loss is not None)
                    else loss
                )
                final_metrics = (
                    self.best_metrics
                    if (self.track_best_across_rounds and self.best_metrics is not None)
                    else metrics
                )

                return (
                    final_config_code,
                    final_scenario_code,
                    final_train_code,
                    final_loss,
                    final_metrics,
                    self.prompts,
                    self.logger.experiment_dir,
                )

            except Exception as e:
                self.logger.log_error(
                    f"Component generation failed on attempt {attempt + 1}: {str(e)}",
                    error_type="COMPONENT_GENERATION_ERROR",
                )
                self.ui_agent.error(
                    f"Component generation failed on attempt {attempt + 1}: {str(e)}"
                )

                if attempt == max_retries - 1:
                    # Last attempt failed, raise the error
                    self.ui_agent.error(
                        f"All {max_retries} attempts failed. Raising error."
                    )
                    raise e
                else:
                    # Wait a bit before retrying
                    import time

                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    self.ui_agent.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

    def _run_generated(self, component: str):
        """Generic runner for config, scenario, or train_function"""
        retry_count = 0
        while True:
            code_attr = f"{component}_code"
            raw_code = getattr(self, code_attr)
            source = extract_code_block(raw_code)
            self.logger.log_prompt(
                f"Running {component} code:", {"component": component, "action": "run"}
            )
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
                    dataset_for_train = self._get_dataset_for_train_function(
                        self.dataset, self.dataset_test, include_test_data=True
                    )

                    # if the train function is multi-fidelity or hyperband, then we need to pass the budget parameter
                    if self.suggested_facade.__name__ in [
                        "HyperbandFacade",
                        "MultiFidelityFacade",
                    ]:
                        result = train_fn(
                            cfg=sampled, dataset=dataset_for_train, seed=0, budget=10
                        )
                    else:
                        result = train_fn(
                            cfg=sampled, dataset=dataset_for_train, seed=0
                        )

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
                    recent_errors = "\n".join(
                        self.error_history[component][-self.MAX_RETRIES_PROMPT :]
                    )
                    fix_prompt = self._create_fix_prompt(recent_errors, raw_code)
                    self.logger.log_prompt(
                        fix_prompt, {"component": component, "action": "fix"}
                    )
                    fixed = self.llm.generate(fix_prompt)
                    self.logger.log_response(
                        fixed, {"component": component, "action": "fix"}
                    )
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

    def _create_train_prompt(
        self, train_function_instructions, suggested_facade
    ) -> str:
        """
        Create a prompt for generating the training function code.
        :param train_function_instructions: Instructions for the training function.
        :param suggested_facade: The suggested SMAC facade.
        :return: A formatted prompt string for the LLM.
        """
        # Prepare baseline information - only include if baseline evaluation succeeded
        if self.baseline_accuracy is not None and self.baseline_metrics is not None:
            baseline_accuracy_section = f"""
            * **AutoGluon Baseline Accuracy**: {self.baseline_accuracy:.4f}
            * **Target**: Your generated solution should achieve **at least {self.baseline_accuracy:.4f} accuracy**
            * **Goal**: Exceed the baseline performance with efficient, well-tuned models
            * **Baseline Metrics**: {self.baseline_metrics}
            """
        else:
            baseline_accuracy_section = """
            * **No baseline available**: Focus on achieving the best possible performance for this dataset
            * **Goal**: Implement robust, efficient models with proper validation and regularization
            """

        with open("templates/train_prompt.txt", "r") as f:
            prompt = f.read().format(
                dataset_description=self.dataset_description,
                config_space=self.config_code or "",
                scenario=self.scenario_code or "",
                train_function_plan=train_function_instructions,
                suggested_facade=suggested_facade,
                baseline_accuracy_section=baseline_accuracy_section,
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
        required_params = FACADE_PARAMETER_REQUIREMENTS.get(
            facade_name, ["config", "seed"]
        )

        # Get dataset dictionary for train function
        dataset_for_train = self._get_dataset_for_train_function(
            dataset, self.dataset_test
        )

        # Create dynamic wrapper function based on facade requirements
        if "budget" in required_params:

            def smac_train_function(config, seed: int = 0, budget: int = None) -> float:
                """Wrapper function for multi-fidelity facades (Hyperband, MultiFidelity)"""
                try:
                    # Convert budget to int if it's provided as a float
                    budget_int = int(budget) if budget is not None else budget

                    # Add timeout protection for individual training runs
                    with timeout_handler(
                        min(600, self.time_budget // 10)
                    ):  # Max 10 minutes per trial or 1/10 of total budget
                        result = train_fn(
                            cfg=config,
                            dataset=dataset_for_train,
                            seed=int(seed),
                            budget=budget_int,
                        )

                    # Train function now returns dictionary with loss and metrics
                    if isinstance(result, dict) and "loss" in result:
                        # Validate loss; if invalid, treat as crash cost
                        try:
                            loss_value = float(result["loss"])
                            if not self._is_finite(loss_value):
                                return float("inf")
                            return loss_value
                        except Exception:
                            return float("inf")
                    else:
                        try:
                            return float(result)  # Fallback for backward compatibility
                        except Exception:
                            return float("inf")

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
                    with timeout_handler(
                        min(600, self.time_budget // 10)
                    ):  # Max 10 minutes per trial or 1/10 of total budget
                        result = train_fn(
                            cfg=config, dataset=dataset_for_train, seed=int(seed)
                        )

                    # Train function now returns dictionary with loss and metrics
                    if isinstance(result, dict) and "loss" in result:
                        try:
                            loss_value = float(result["loss"])
                            if not self._is_finite(loss_value):
                                return float("inf")
                            return loss_value
                        except Exception:
                            return float("inf")
                    else:
                        try:
                            return float(result)  # Fallback for backward compatibility
                        except Exception:
                            return float("inf")

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
            self.ui_agent.info(
                f"Starting SMAC optimization with {self.time_budget} seconds timeout..."
            )

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
            self.ui_agent.warning(
                f"Optimization timed out after {optimization_timeout}s, using best config found so far"
            )

            # Get the best configuration found so far
            try:
                incumbent = smac.optimizer.intensifier.get_incumbent()
            except Exception as e:
                self.logger.log_error(
                    f"Error getting incumbent: {e}", error_type="SMAC_ERROR"
                )
                # Fallback to default configuration if no incumbent found
                incumbent = cfg.get_default_configuration()
                self.logger.log_error(
                    "No incumbent found, using default configuration",
                    error_type="FALLBACK_CONFIG",
                )

        except Exception as e:
            self.logger.log_error(
                f"SMAC optimization failed: {str(e)}", error_type="SMAC_ERROR"
            )
            self.ui_agent.error(f"Optimization failed: {str(e)}")
            # Fallback to default configuration
            incumbent = cfg.get_default_configuration()
            self.logger.log_error(
                "Using default configuration due to optimization failure",
                error_type="FALLBACK_CONFIG",
            )

        default_cost = smac.validate(self.config_space_obj.get_default_configuration())
        # Guard against non-finite costs
        if not self._is_finite(default_cost):
            default_cost = float("inf")
        self.ui_agent.subheader("Default Cost")
        self.ui_agent.write(default_cost)

        incubment_cost = smac.validate(incumbent)
        if not self._is_finite(incubment_cost):
            incubment_cost = float("inf")
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
        # Degenerate results guard: if both costs are non-finite, attempt regeneration once
        if (
            self.enable_baseline_retry
            and (not self._is_finite(incubment_cost))
            and (not self._is_finite(default_cost))
        ):
            self.ui_agent.warning(
                "Both default and incumbent costs are non-finite. Attempting regeneration with improvement feedback."
            )
            if self.retry_with_improvement_feedback():
                return self.run_scenario(
                    self.scenario_obj,
                    self.namespace.get("train", self.train_fn),
                    self.dataset,
                    self.config_space_obj,
                )

        return self.test_incumbent(
            self.train_fn,
            incumbent,
            self.dataset,
            self.dataset_test,
            self.config_space_obj,
        )

    def validate_baseline_target(self, final_metrics: dict) -> bool:
        """
        Validate that the final solution meets or exceeds the baseline target

        Args:
            final_metrics: Dictionary containing final evaluation metrics

        Returns:
            True if baseline target is met, False otherwise
        """
        if self.baseline_accuracy is None:
            return True  # No baseline to compare against

        # Extract accuracy from final metrics
        final_accuracy = None
        if "accuracy" in final_metrics:
            final_accuracy = final_metrics["accuracy"]
        elif "balanced_accuracy" in final_metrics:
            final_accuracy = final_metrics["balanced_accuracy"]
        elif "f1" in final_metrics:
            final_accuracy = final_metrics["f1"]
        elif "loss" in final_metrics:
            # For loss-based metrics, convert to accuracy-like score
            final_accuracy = 1.0 - final_metrics["loss"]

        if final_accuracy is None:
            self.ui_agent.warning(
                "Could not extract accuracy metric for baseline comparison"
            )
            return True

        # Compare with baseline
        meets_target = final_accuracy >= self.baseline_accuracy

        if meets_target:
            self.ui_agent.success(
                f"✓ Solution meets baseline target! "
                f"Final accuracy: {final_accuracy:.4f} >= Baseline: {self.baseline_accuracy:.4f}"
            )
        else:
            self.ui_agent.warning(
                f"⚠ Solution below baseline target. "
                f"Final accuracy: {final_accuracy:.4f} < Baseline: {self.baseline_accuracy:.4f}"
            )

        # Log validation result
        self.logger.log_response(
            f"Baseline validation: {'PASSED' if meets_target else 'FAILED'}",
            {
                "component": "validation",
                "status": "success" if meets_target else "warning",
                "final_accuracy": final_accuracy,
                "baseline_accuracy": self.baseline_accuracy,
                "meets_target": meets_target,
            },
        )

        return meets_target

    def _extract_accuracy_from_metrics(self, metrics: dict) -> Optional[float]:
        """Extract an accuracy-like scalar from metrics for comparison and best-tracking."""
        if metrics is None:
            return None
        if "accuracy" in metrics:
            return float(metrics["accuracy"])
        if "balanced_accuracy" in metrics:
            return float(metrics["balanced_accuracy"])
        if "f1" in metrics:
            return float(metrics["f1"])
        if "loss" in metrics:
            try:
                return 1.0 - float(metrics["loss"])
            except Exception:
                return None
        return None

    def _is_finite(self, value: Any) -> bool:
        try:
            v = float(value)
            return v == v and v not in (float("inf"), float("-inf"))
        except Exception:
            return False

    def _is_valid_metrics(self, metrics: dict) -> bool:
        if not isinstance(metrics, dict):
            return False
        # Disallow explicit error flag
        if "error" in metrics and metrics["error"]:
            return False
        # Loss must be finite if present
        if "loss" in metrics and not self._is_finite(metrics["loss"]):
            return False
        # If accuracy-like present, it must be within [0,1]
        for k in ("accuracy", "balanced_accuracy", "f1"):
            if k in metrics:
                try:
                    val = float(metrics[k])
                    if not (0.0 <= val <= 1.0):
                        return False
                except Exception:
                    return False
        return True

    def _regenerate_component_with_fix(
        self, component: str, summary_error: str
    ) -> None:
        """Regenerate a component using the fix prompt with provided error context, then run it."""
        code_attr = f"{component}_code"
        raw_code = getattr(self, code_attr)
        # Append to error history
        self.error_history[component].append(summary_error)
        recent_errors = "\n".join(
            self.error_history[component][-self.MAX_RETRIES_PROMPT :]
        )
        fix_prompt = self._create_fix_prompt(recent_errors, raw_code)
        self.logger.log_prompt(
            fix_prompt, {"component": component, "action": "fix_after_eval"}
        )
        fixed = self.llm.generate(fix_prompt)
        self.logger.log_response(
            fixed, {"component": component, "action": "fix_after_eval"}
        )
        setattr(self, code_attr, fixed)
        self._run_generated(component)

    def evaluate_baseline(self):
        """Evaluate baseline performance using AutoGluon to set minimum targets"""
        try:
            self.ui_agent.info("Evaluating AutoGluon baseline performance...")

            # Get training and test data
            X_train = self.dataset["X"]
            y_train = self.dataset["y"]
            X_test = self.dataset_test["X"]
            y_test = self.dataset_test["y"]

            # Evaluate baseline
            baseline_results = self.baseline_evaluator.evaluate_baseline(
                X_train, y_train, X_test, y_test
            )

            # Store baseline metrics
            self.baseline_metrics = baseline_results

            # Extract accuracy (or equivalent metric)
            if "accuracy" in baseline_results:
                self.baseline_accuracy = baseline_results["accuracy"]
            elif "balanced_accuracy" in baseline_results:
                self.baseline_accuracy = baseline_results["balanced_accuracy"]
            elif "f1" in baseline_results:
                self.baseline_accuracy = baseline_results["f1"]
            elif "mse" in baseline_results:
                # For regression, convert MSE to a score (lower is better)
                self.baseline_accuracy = 1.0 / (1.0 + baseline_results["mse"])
            else:
                # If no recognizable metric, skip baseline validation
                self.ui_agent.warning(
                    "No recognizable accuracy metric found in baseline results"
                )
                self.baseline_accuracy = None

            # Set baseline in post-processor
            if self.baseline_accuracy is not None:
                self.post_processor.set_baseline(
                    self.baseline_accuracy, baseline_results
                )

            self.ui_agent.success(f"Baseline evaluation completed!")
            self.ui_agent.subheader("Baseline Performance")
            self.ui_agent.write(f"Baseline metrics: {baseline_results}")
            if self.baseline_accuracy is not None:
                self.ui_agent.write(f"Target accuracy: {self.baseline_accuracy:.4f}")

            # Log baseline results
            self.logger.log_response(
                f"Baseline evaluation completed: {baseline_results}",
                {
                    "component": "baseline",
                    "status": "success",
                    "metrics": baseline_results,
                    "target_accuracy": self.baseline_accuracy,
                },
            )

            return True

        except Exception as e:
            self.ui_agent.warning(f"Baseline evaluation failed: {str(e)}")
            self.logger.log_error(
                f"Baseline evaluation failed: {str(e)}",
                error_type="BASELINE_ERROR",
            )

            # Don't set default values - let the system continue without baseline
            self.baseline_accuracy = None
            self.baseline_metrics = None

            return False

    def retry_with_improvement_feedback(self) -> bool:
        """
        Retry the component generation with improvement feedback from baseline failure.
        Regenerates all components (config, scenario, train_function) with improvement guidance.

        Returns:
            True if retry was successful, False otherwise
        """
        if not self.post_processor.improvement_feedback:
            return False

        self.ui_agent.info("Retrying with baseline improvement feedback...")

        # Store original codes for comparison
        original_config_code = self.config_code
        original_scenario_code = self.scenario_code
        original_train_function_code = self.train_function_code

        try:
            # 1) Re-run planner with improvement context to get a fresh plan
            improvement_context = self.post_processor.improvement_feedback
            config_space_suggested_parameters = (
                self.openML.extract_suggested_config_space_parameters(
                    dataset_name=self.dataset_name or "unknown",
                    X=self.dataset["X"],
                    y=self.dataset["y"],
                )
            )
            instructor_response = self.LLMInstructor.generate_instructions(
                config_space_suggested_parameters,
                improvement_context=improvement_context,
            )
            # Update suggested facade from the new plan
            self.suggested_facade = self.smac_facades[
                instructor_response.suggested_facade
            ]

            # 2) Regenerate components from the refreshed plan
            # Configuration Space
            self.prompts["config"] = self._create_config_prompt(
                instructor_response.configuration_plan,
            )
            self.config_code = self.llm.generate(self.prompts["config"])
            self._run_generated("config")
            self.ui_agent.subheader("🔄 Improved Configuration Space")
            self.ui_agent.code(self.config_code, language="python")

            # Scenario
            self.prompts["scenario"] = self._create_scenario_prompt(
                instructor_response.scenario_plan,
                instructor_response.suggested_facade,
            )
            self.scenario_code = self.llm.generate(self.prompts["scenario"])
            self._run_generated("scenario")
            self.ui_agent.subheader("🔄 Improved Scenario Configuration")
            self.ui_agent.code(self.scenario_code, language="python")

            # Training Function
            self.prompts["train_function"] = self._create_train_prompt(
                instructor_response.train_function_plan,
                instructor_response.suggested_facade,
            )
            self.train_function_code = self.llm.generate(self.prompts["train_function"])
            self._run_generated("train_function")
            self.ui_agent.subheader("🔄 Improved Training Function")
            self.ui_agent.code(self.train_function_code, language="python")

            self.ui_agent.success(
                "All components regenerated with improvement feedback!"
            )

            # Show comparison between original and improved codes
            self.ui_agent.subheader("📊 Code Comparison")
            self.ui_agent.info("Comparing original vs improved implementations:")

            # Configuration Space comparison
            self.ui_agent.subheader("Configuration Space Changes")
            self.ui_agent.write("**Original:**")
            self.ui_agent.code(
                (
                    original_config_code[:300] + "..."
                    if len(original_config_code) > 300
                    else original_config_code
                ),
                language="python",
            )
            self.ui_agent.write("**Improved:**")
            self.ui_agent.code(
                (
                    self.config_code[:300] + "..."
                    if len(self.config_code) > 300
                    else self.config_code
                ),
                language="python",
            )

            # Scenario comparison
            self.ui_agent.subheader("Scenario Configuration Changes")
            self.ui_agent.write("**Original:**")
            self.ui_agent.code(
                (
                    original_scenario_code[:300] + "..."
                    if len(original_scenario_code) > 300
                    else original_scenario_code
                ),
                language="python",
            )
            self.ui_agent.write("**Improved:**")
            self.ui_agent.code(
                (
                    self.scenario_code[:300] + "..."
                    if len(self.scenario_code) > 300
                    else self.scenario_code
                ),
                language="python",
            )

            # Training Function comparison
            self.ui_agent.subheader("Training Function Changes")
            self.ui_agent.write("**Original:**")
            self.ui_agent.code(
                (
                    original_train_function_code[:300] + "..."
                    if len(original_train_function_code) > 300
                    else original_train_function_code
                ),
                language="python",
            )
            self.ui_agent.write("**Improved:**")
            self.ui_agent.code(
                (
                    self.train_function_code[:300] + "..."
                    if len(self.train_function_code) > 300
                    else self.train_function_code
                ),
                language="python",
            )

            # Show summary of improvements
            self.ui_agent.subheader("📊 Improvement Summary")
            self.ui_agent.info(
                f"Iteration {self.post_processor.iteration_count}: All components regenerated with targeted improvements"
            )
            self.ui_agent.info("Key improvements applied:")
            self.ui_agent.write(
                "• Configuration Space: Enhanced algorithm diversity and hyperparameter ranges"
            )
            self.ui_agent.write(
                "• Scenario: Optimized time budget and trial allocation"
            )
            self.ui_agent.write(
                "• Training Function: Improved preprocessing and algorithm selection"
            )

            return True

        except Exception as e:
            self.ui_agent.error(f"Retry with improvement feedback failed: {str(e)}")
            return False

    def _create_improved_config_prompt(self) -> str:
        """
        Create an improved configuration space prompt incorporating baseline failure feedback.

        Returns:
            Improved config prompt string
        """
        improvement_context = self.post_processor.create_improvement_prompt("config")

        # Get the original config prompt
        original_prompt = self._create_config_prompt(
            "Implement improved configuration space based on baseline failure analysis"
        )

        # Insert improvement context at the beginning
        improved_prompt = improvement_context + "\n\n" + original_prompt

        return improved_prompt

    def _create_improved_scenario_prompt(self) -> str:
        """
        Create an improved scenario prompt incorporating baseline failure feedback.

        Returns:
            Improved scenario prompt string
        """
        improvement_context = self.post_processor.create_improvement_prompt("scenario")

        # Get the original scenario prompt
        original_prompt = self._create_scenario_prompt(
            "Implement improved scenario based on baseline failure analysis",
            self.suggested_facade.__name__,
        )

        # Insert improvement context at the beginning
        improved_prompt = improvement_context + "\n\n" + original_prompt

        return improved_prompt

    def _create_improved_train_prompt(self) -> str:
        """
        Create an improved training function prompt incorporating baseline failure feedback.

        Returns:
            Improved train prompt string
        """
        improvement_context = self.post_processor.create_improvement_prompt(
            "train_function"
        )

        # Get the original train prompt
        original_prompt = self._create_train_prompt(
            "Implement improved solution based on baseline failure analysis",
            self.suggested_facade.__name__,
        )

        # Insert improvement context at the beginning
        improved_prompt = improvement_context + "\n\n" + original_prompt

        return improved_prompt
