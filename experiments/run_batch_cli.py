"""
Command-line batch runner for multiple datasets.

This script runs multiple datasets through the AutoML Agent in batch mode
without requiring Streamlit UI, suitable for command-line execution.
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback  # noqa: E402
import time  # noqa: E402
from typing import List, Dict, Any  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402

from scripts.LLMClient import LLMClient  # noqa: E402
from scripts.AutoMLAgent import AutoMLAgent  # noqa: E402
from config.api_keys import GOOGLE_API_KEY  # noqa: E402


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


def run_batch_experiments_cli(datasets: List[Dict[str, Any]], model_name: str = "gemini-2.0-flash"):
    """
    Run batch experiments on multiple datasets using CLI interface.

    Args:
        datasets: List of dataset configurations
        model_name: Name of the LLM model to use
    """
    # Initialize CLI UI
    cli_ui = CLIUI()

    # Initialize LLM client
    cli_ui.info(f"Initializing LLM client with model: {model_name}")
    llm_client = LLMClient(model_name=model_name, api_key=GOOGLE_API_KEY)

    # Results storage
    results = []

    # Process each dataset
    for i, dataset_config in enumerate(datasets, 1):
        dataset_name = dataset_config["dataset_name"]
        dataset_type = dataset_config["dataset_type"]
        task_type = dataset_config["task_type"]

        cli_ui.subheader(f"Processing Dataset {i}/{len(datasets)}: {dataset_name}")
        cli_ui.info(f"Type: {dataset_type}, Task: {task_type}")

        start_time = time.time()

        try:
            # Load dataset from OpenML
            cli_ui.info(f"Loading dataset '{dataset_name}' from OpenML...")
            X, y = load_dataset_by_name(dataset_name)
            dataset_df = prepare_dataset_for_automl(X, y, dataset_name)
            cli_ui.success(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

            # Initialize AutoML Agent
            cli_ui.info("Initializing AutoML Agent...")
            agent = AutoMLAgent(
                dataset=dataset_df,
                llm_client=llm_client,
                dataset_type=dataset_type,
                ui_agent=cli_ui,
                dataset_name=dataset_name,
                task_type=task_type,
            )

            # Generate components and run optimization
            cli_ui.info("Generating components and running optimization...")
            agent_results = agent.generate_components()

            # Calculate duration
            duration = time.time() - start_time

            # Store results
            result = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "task_type": task_type,
                "status": "success",
                "duration_seconds": duration,
                "config_code": agent_results[0],
                "scenario_code": agent_results[1],
                "train_function_code": agent_results[2],
                "last_loss": agent_results[3],
                "experiment_dir": agent_results[5],
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

            cli_ui.success(f"Dataset {dataset_name} completed successfully in {duration:.1f}s")

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Store error result
            error_message = f"Error processing {dataset_name}: {str(e)}"
            result = {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "task_type": task_type,
                "status": "error",
                "duration_seconds": duration,
                "error_message": error_message,
            }
            results.append(result)

            cli_ui.error(f"Dataset {dataset_name} failed after {duration:.1f}s: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            continue

    # Final summary
    cli_ui.subheader("Batch Processing Summary")

    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "error"])
    total_duration = sum(r["duration_seconds"] for r in results)

    cli_ui.info(f"Total datasets: {len(datasets)}")
    cli_ui.success(f"Successful: {successful}")
    cli_ui.error(f"Failed: {failed}")
    cli_ui.info(f"Total duration: {total_duration:.1f}s")

    # Display detailed results
    cli_ui.subheader("Detailed Results")
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        duration = result["duration_seconds"]

        if result["status"] == "success":
            test_acc = result.get("test_train_accuracy", "N/A")
            cli_ui.write(f"{status_emoji} {result['dataset_name']}: {duration:.1f}s, Test Acc: {test_acc}")
        else:
            cli_ui.write(f"{status_emoji} {result['dataset_name']}: {duration:.1f}s, Error: {result['error_message'][:100]}...")

    # Save results to CSV
    try:
        df = pd.DataFrame(results)
        output_file = f"batch_results_{int(time.time())}.csv"
        df.to_csv(output_file, index=False)
        cli_ui.success(f"Results saved to: {output_file}")
    except Exception as e:
        cli_ui.error(f"Failed to save results: {e}")

    return results


# OpenML Dataset configurations
openml_datasets = [
    {
        "dataset_name": "iris",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "wine",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "breast_cancer",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "diabetes",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
    {
        "dataset_name": "vehicle",
        "dataset_type": "tabular",
        "task_type": "classification",
        "model_name": "gemini-2.0-flash",
        "seed": 42,
    },
]


if __name__ == "__main__":
    print("🤖 AutoML Agent - Batch Processing (CLI Mode)")
    print("=" * 60)
    print("Using OpenML datasets via sklearn.fetch_openml")
    print("=" * 60)

    # Run batch experiments
    results = run_batch_experiments_cli(openml_datasets)

    print("\n" + "=" * 60)
    print("Batch processing completed!")
    print("=" * 60)
