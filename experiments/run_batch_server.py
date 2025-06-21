"""
Server-optimized batch runner for multiple datasets.

This script runs multiple datasets through the AutoML Agent in batch mode
without any UI dependencies, optimized for server deployment.
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback  # noqa: E402
import time  # noqa: E402
import logging  # noqa: E402
from typing import List, Dict, Any  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402

from scripts.LLMClient import LLMClient  # noqa: E402
from scripts.AutoMLAgent import AutoMLAgent  # noqa: E402
from config.api_keys import GOOGLE_API_KEY  # noqa: E402


class ServerUI:
    """Minimal UI agent for server deployment - logs everything to files."""

    def __init__(self, dataset_name: str, log_dir: str = "./logs/server"):
        """Initialize server UI with logging."""
        self.dataset_name = dataset_name
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"server_ui_{dataset_name}")
        self.logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(log_dir, f"{dataset_name}_server.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

        # Also log to console for server monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def subheader(self, text: str):
        """Log a subheader."""
        self.logger.info(f"=== {text} ===")

    def write(self, text: Any):
        """Log text."""
        self.logger.info(f"INFO: {text}")

    def code(self, code: str, language: str = "python"):
        """Log code (truncated for brevity)."""
        truncated_code = code[:500] + "..." if len(code) > 500 else code
        self.logger.info(f"CODE ({language}): {truncated_code}")

    def success(self, message: str):
        """Log success message."""
        self.logger.info(f"SUCCESS: {message}")

    def error(self, message: str):
        """Log error message."""
        self.logger.error(f"ERROR: {message}")

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(f"WARNING: {message}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(f"INFO: {message}")


def load_dataset_by_name(dataset_name: str):
    """
    Load a dataset by name from OpenML.

    Args:
        dataset_name: Name of the dataset to load from OpenML

    Returns:
        Tuple of (X, y) data
    """
    try:
        # Fetch dataset from OpenML
        dataset = fetch_openml(name=dataset_name, as_frame=True, parser="auto")

        # Extract features and target
        X = dataset.data
        y = dataset.target

        # Convert target to numeric if it's categorical
        if y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(y)

        return X, y

    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}' from OpenML: {e}")


def prepare_dataset_for_automl(X, y, dataset_name: str):
    """
    Prepare dataset in the format expected by AutoML Agent.

    Args:
        X: Features
        y: Target
        dataset_name: Name of the dataset

    Returns:
        DataFrame with features and target column
    """
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Add target column
    df = X.copy()
    df["target"] = y

    return df


def run_batch_experiments_server(
    datasets: List[Dict[str, Any]],
    model_name: str = "gemini-2.0-flash",
    output_dir: str = "./batch_results",
):
    """
    Run batch experiments on multiple datasets optimized for server deployment.

    Args:
        datasets: List of dataset configurations
        model_name: Name of the LLM model to use
        output_dir: Directory to save results

    Returns:
        List of results dictionaries
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup main logging
    main_logger = logging.getLogger("batch_server")
    main_logger.setLevel(logging.INFO)

    # Create main log file
    main_log_file = os.path.join(output_dir, f"batch_run_{int(time.time())}.log")
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    main_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    main_logger.addHandler(console_handler)

    main_logger.info("🤖 AutoML Agent - Server Batch Processing Started")
    main_logger.info(f"Processing {len(datasets)} datasets with model: {model_name}")
    main_logger.info("Using OpenML datasets via sklearn.fetch_openml")

    # Initialize LLM client
    main_logger.info("Initializing LLM client...")
    llm_client = LLMClient(model_name=model_name, api_key=GOOGLE_API_KEY)

    # Results storage
    results = []
    start_time = time.time()

    # Process each dataset
    for i, dataset_config in enumerate(datasets, 1):
        time.sleep(60)  # sleep for 60 seconds to avoid rate limiting
        dataset_name = dataset_config["dataset_name"]
        dataset_type = dataset_config["dataset_type"]
        task_type = dataset_config["task_type"]

        main_logger.info(f"Processing dataset {i}/{len(datasets)}: {dataset_name}")
        main_logger.info(f"Type: {dataset_type}, Task: {task_type}")

        dataset_start_time = time.time()

        try:
            # Initialize server UI for this dataset
            server_ui = ServerUI(dataset_name, output_dir)

            # Load dataset from OpenML
            server_ui.info(f"Loading dataset '{dataset_name}' from OpenML...")
            X, y = load_dataset_by_name(dataset_name)
            dataset_df = prepare_dataset_for_automl(X, y, dataset_name)
            server_ui.success(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

            # Initialize AutoML Agent
            server_ui.info("Initializing AutoML Agent...")
            agent = AutoMLAgent(
                dataset=dataset_df,
                llm_client=llm_client,
                dataset_type=dataset_type,
                ui_agent=server_ui,
                dataset_name=dataset_name,
                task_type=task_type,
            )

            # Generate components and run optimization
            server_ui.info("Generating components and running optimization...")
            agent_results = agent.generate_components()

            # Calculate duration
            dataset_duration = time.time() - dataset_start_time

            # Store results
            result = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "task_type": task_type,
                "status": "success",
                "duration_seconds": dataset_duration,
                "config_code": agent_results[0],
                "scenario_code": agent_results[1],
                "train_function_code": agent_results[2],
                "last_loss": agent_results[3],
                "experiment_dir": agent_results[5],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add experimenter results if available
            if hasattr(agent, "experimenter_row_dict"):
                result.update(
                    {
                        "default_train_accuracy": agent.experimenter_row_dict.get("default_train_accuracy"),
                        "incumbent_train_accuracy": agent.experimenter_row_dict.get("incumbent_train_accuracy"),
                        "test_train_accuracy": agent.experimenter_row_dict.get("test_train_accuracy"),
                        "incumbent_config": str(agent.experimenter_row_dict.get("incumbent_config", "")),
                    }
                )

            results.append(result)

            server_ui.success(f"Dataset {dataset_name} completed successfully in {dataset_duration:.1f}s")
            main_logger.info(f"✅ {dataset_name}: SUCCESS ({dataset_duration:.1f}s)")

        except Exception as e:
            # Calculate duration
            dataset_duration = time.time() - dataset_start_time

            # Store error result
            error_message = f"Error processing {dataset_name}: {str(e)}"
            result = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "task_type": task_type,
                "status": "error",
                "duration_seconds": dataset_duration,
                "error_message": error_message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results.append(result)

            main_logger.error(f"❌ {dataset_name}: FAILED ({dataset_duration:.1f}s) - {e}")
            main_logger.error(f"Full traceback: {traceback.format_exc()}")
            continue

    # Calculate total duration
    total_duration = time.time() - start_time

    # Final summary
    main_logger.info("=" * 60)
    main_logger.info("BATCH PROCESSING SUMMARY")
    main_logger.info("=" * 60)

    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "error"])

    main_logger.info(f"Total datasets: {len(datasets)}")
    main_logger.info(f"Successful: {successful}")
    main_logger.info(f"Failed: {failed}")
    main_logger.info(f"Total duration: {total_duration:.1f}s")
    main_logger.info(f"Average per dataset: {total_duration/len(datasets):.1f}s")

    # Save detailed results to CSV
    try:
        df = pd.DataFrame(results)
        csv_file = os.path.join(output_dir, f"batch_results_{int(time.time())}.csv")
        df.to_csv(csv_file, index=False)
        main_logger.info(f"Results saved to: {csv_file}")
    except Exception as e:
        main_logger.error(f"Failed to save results: {e}")

    # Save summary to JSON for easy parsing
    try:
        import json

        summary = {
            "total_datasets": len(datasets),
            "successful": successful,
            "failed": failed,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / len(datasets),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }

        json_file = os.path.join(output_dir, f"batch_summary_{int(time.time())}.json")
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)
        main_logger.info(f"Summary saved to: {json_file}")
    except Exception as e:
        main_logger.error(f"Failed to save summary: {e}")

    main_logger.info("=" * 60)
    main_logger.info("Batch processing completed!")
    main_logger.info("=" * 60)

    return results


# OpenML Dataset configurations
openml_datasets = [
    {
        "dataset_name": "credit-g",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "adult",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "iris",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "Sick",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
]


if __name__ == "__main__":
    # Run batch experiments
    results = run_batch_experiments_server(openml_datasets)

    # Exit with appropriate code
    successful = len([r for r in results if r["status"] == "success"])
    total = len(openml_datasets)

    if successful == total:
        print(f"✅ All {total} datasets completed successfully!")
        exit(0)
    else:
        print(f"⚠️  {successful}/{total} datasets completed successfully. {total-successful} failed.")
        exit(1)
