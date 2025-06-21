"""
Batch runner for multiple datasets using Streamlit UI.

This script runs multiple datasets through the AutoML Agent in batch mode
with a Streamlit-based UI for monitoring progress and viewing results.
"""

import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import List, Dict, Any  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402

from scripts.LLMClient import LLMClient  # noqa: E402
from scripts.AutoMLAgent import AutoMLAgent  # noqa: E402
from scripts.BatchUI import BatchUI  # noqa: E402
from config.api_keys import GOOGLE_API_KEY  # noqa: E402


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


def run_batch_experiments(datasets: List[Dict[str, Any]], model_name: str = "gemini-2.0-flash"):
    """
    Run batch experiments on multiple datasets.

    Args:
        datasets: List of dataset configurations
        model_name: Name of the LLM model to use
    """
    # Initialize BatchUI
    batch_ui = BatchUI()

    # Initialize LLM client
    llm_client = LLMClient(model_name=model_name, api_key=GOOGLE_API_KEY)

    # Process each dataset
    for dataset_config in datasets:
        dataset_name = dataset_config["dataset_name"]
        dataset_type = dataset_config["dataset_type"]
        task_type = dataset_config["task_type"]

        try:
            # Add dataset to batch UI
            batch_ui.add_dataset(dataset_name, dataset_type, task_type)

            # Load dataset
            X, y = load_dataset_by_name(dataset_name)
            dataset_df = prepare_dataset_for_automl(X, y, dataset_name)

            # Initialize AutoML Agent
            agent = AutoMLAgent(
                dataset=dataset_df,
                llm_client=llm_client,
                dataset_type=dataset_type,
                ui_agent=batch_ui,  # Use batch UI instead of regular UI
                dataset_name=dataset_name,
                task_type=task_type,
            )

            # Generate components and run optimization
            results = agent.generate_components()

            # Update batch UI with results
            batch_ui.update_dataset_status(
                dataset_name=dataset_name,
                status="success",
                config_code=results[0],
                scenario_code=results[1],
                train_function_code=results[2],
                last_loss=results[3],
                experiment_dir=results[5],
            )

            # Update experimenter row dict with results
            if hasattr(agent, "experimenter_row_dict"):
                batch_ui.update_dataset_status(
                    dataset_name=dataset_name,
                    default_train_accuracy=agent.experimenter_row_dict.get("default_train_accuracy"),
                    incumbent_train_accuracy=agent.experimenter_row_dict.get("incumbent_train_accuracy"),
                    test_train_accuracy=agent.experimenter_row_dict.get("test_train_accuracy"),
                    incumbent_config=agent.experimenter_row_dict.get("incumbent_config"),
                )

        except Exception as e:
            # Update batch UI with error
            error_message = f"Error processing {dataset_name}: {str(e)}\n{traceback.format_exc()}"
            batch_ui.update_dataset_status(dataset_name=dataset_name, status="error", error_message=error_message)
            print(f"Error processing {dataset_name}: {e}")
            continue

    # Final summary
    batch_ui.success("Batch processing completed!")
    return batch_ui


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
    # Run batch experiments
    batch_ui = run_batch_experiments(openml_datasets)
