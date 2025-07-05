"""
Training utilities for AutoML Agent.

This module provides functions for executing training functions,
handling their results, and managing training parameters.
"""

from typing import Any, Tuple, Optional, Dict
import logging


def execute_training_function(
    train_fn: Any,
    cfg: Any,
    dataset: Any,
    seed: int = 0,
    budget: Optional[int] = None,
    model_dir: Optional[str] = None,
) -> Tuple[float, Optional[Any]]:
    """
    Execute a training function and handle its return value.

    Args:
        train_fn: The training function to execute
        cfg: Configuration object for the training function
        dataset: Dataset for training
        seed: Random seed for reproducibility
        budget: Optional budget parameter for multi-fidelity optimization
        model_dir: Optional directory to save the model

    Returns:
        Tuple[float, Optional[Any]]: (loss, model) where model can be None
    """
    try:
        # Build arguments for training function
        kwargs = {
            "cfg": cfg,
            "dataset": dataset,
            "seed": seed,
        }

        if budget is not None:
            kwargs["budget"] = budget
        if model_dir is not None:
            kwargs["model_dir"] = model_dir

        # Execute training function
        result = train_fn(**kwargs)

        # Handle different return types
        if isinstance(result, tuple) and len(result) == 2:
            # Training function returned (loss, model)
            loss, model = result
            return float(loss), model
        elif isinstance(result, (int, float)):
            # Training function returned only loss
            return float(result), None
        else:
            # Unexpected return type
            logging.warning(f"Training function returned unexpected type: {type(result)}")
            return float(result), None

    except Exception as e:
        logging.error(f"Error executing training function: {str(e)}")
        raise


def validate_budget_parameter(budget: Any) -> Optional[int]:
    """
    Validate and convert budget parameter to integer.

    Args:
        budget: Budget parameter (can be int, float, or None)

    Returns:
        Optional[int]: Validated budget as integer, or None
    """
    if budget is None:
        return None

    try:
        return int(budget)
    except (ValueError, TypeError):
        logging.warning(f"Invalid budget parameter: {budget}, using None")
        return None


def prepare_training_parameters(
    facade_name: str,
    required_params: list,
    scenario_obj: Any,
    incumbent: Any,
    dataset: Any,
    seed: int = 0,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare parameters for training function execution.

    Args:
        facade_name: Name of the SMAC facade being used
        required_params: List of required parameters for the facade
        scenario_obj: SMAC scenario object
        incumbent: Best configuration found
        dataset: Training dataset
        seed: Random seed
        model_dir: Optional model save directory

    Returns:
        Dict[str, Any]: Dictionary of prepared parameters
    """
    params = {
        "cfg": incumbent,
        "dataset": dataset,
        "seed": seed,
    }

    if model_dir is not None:
        params["model_dir"] = model_dir

    # Handle budget parameter for multi-fidelity facades
    if "budget" in required_params:
        max_budget = getattr(scenario_obj, "max_budget", None)
        budget_int = validate_budget_parameter(max_budget)
        params["budget"] = budget_int

    return params
