import os
import re
from datetime import datetime
from typing import Optional
import numpy as np


class Logger:
    """
    A class to handle logging of prompts and responses for AutoML experiments.
    """

    def __init__(self, model_name: str, dataset_name: str, base_log_dir: str = "./logs"):
        """
        Initialize the logger with model and dataset information.

        Args:
            model_name (str): Name of the model being used
            dataset_name (str): Name of the dataset being used
            base_log_dir (str): Base directory for storing logs
        """
        # Sanitize names for filesystem compatibility
        self.model_name = self._sanitize_filename(model_name) if model_name else "unknown_model"
        self.dataset_name = self._sanitize_filename(dataset_name) if dataset_name else "unknown_dataset"
        self.base_log_dir = base_log_dir

        # Create log directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            self.base_log_dir,
            f"{self.model_name}_{self.dataset_name}_{timestamp}_{np.random.randint(1000000)}",
        )
        self.prompts_file = os.path.join(self.experiment_dir, "prompts.log")
        self.responses_file = os.path.join(self.experiment_dir, "responses.log")

        # Create directory if it doesn't exist
        try:
            os.makedirs(self.experiment_dir, exist_ok=True)
            # Test write access by creating a small test file
            test_file = os.path.join(self.experiment_dir, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"Logger initialized successfully. Logs will be saved to: {self.experiment_dir}")
        except Exception as e:
            print(f"Warning: Failed to create log directory {self.experiment_dir}: {e}")
            # Fallback to a simpler directory structure
            fallback_dir = os.path.join(self.base_log_dir, f"fallback_{timestamp}")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                self.experiment_dir = fallback_dir
                self.prompts_file = os.path.join(self.experiment_dir, "prompts.log")
                self.responses_file = os.path.join(self.experiment_dir, "responses.log")
                print(f"Using fallback directory: {self.experiment_dir}")
            except Exception as e2:
                print(f"Error: Could not create fallback directory: {e2}")
                # Last resort: use current directory
                self.experiment_dir = "."
                self.prompts_file = f"prompts_{timestamp}.log"
                self.responses_file = f"responses_{timestamp}.log"
                print(f"Using current directory for logs: prompts_{timestamp}.log, responses_{timestamp}.log")

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing or replacing problematic characters.

        Args:
            filename (str): The original filename

        Returns:
            str: Sanitized filename safe for filesystem use
        """
        if not filename:
            return "unknown"

        # Replace problematic characters with underscores
        # Remove or replace: . / \ : * ? " < > |
        sanitized = re.sub(r'[.\/\\:*?"<>|]', "_", filename)
        # Replace multiple consecutive underscores with single underscore
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        # Ensure it's not empty after sanitization
        if not sanitized:
            return "unknown"

        return sanitized

    def _write_log(self, message: str, file_path: str, metadata: Optional[dict] = None):
        """
        Write a log message to the specified file with timestamp and optional metadata.

        Args:
            message (str): The message to log
            file_path (str): Path to the log file
            metadata (dict, optional): Additional metadata to include in the log
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] "

            if metadata:
                log_entry += f"[Metadata: {metadata}] "

            log_entry += f"{message}\n"
            log_entry += "-" * 80 + "\n"  # Add separator line

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(log_entry)

        except Exception as e:
            # Print error to console if logging fails
            print(f"Error writing to log file {file_path}: {e}")
            print(f"Failed to log message: {message[:100]}..." if len(message) > 100 else f"Failed to log message: {message}")

    def log_prompt(self, prompt: str, metadata: Optional[dict] = None):
        """
        Log a prompt message.

        Args:
            prompt (str): The prompt to log
            metadata (dict, optional): Additional metadata about the prompt
        """
        try:
            self._write_log(str(prompt), self.prompts_file, metadata)
        except Exception as e:
            print(f"Error logging prompt: {e}")

    def log_response(self, response: str, metadata: Optional[dict] = None):
        """
        Log a response message.

        Args:
            response (str): The response to log
            metadata (dict, optional): Additional metadata about the response
        """
        try:
            self._write_log(str(response), self.responses_file, metadata)
        except Exception as e:
            print(f"Error logging response: {e}")

    def log_error(self, error_message: str, error_type: str = "ERROR"):
        """
        Log an error message to both prompt and response files.

        Args:
            error_message (str): The error message to log
            error_type (str): Type of error (default: "ERROR")
        """
        try:
            metadata = {"error_type": error_type}
            error_entry = f"[{error_type}] {error_message}"
            self._write_log(error_entry, self.prompts_file, metadata)
            self._write_log(error_entry, self.responses_file, metadata)
        except Exception as e:
            print(f"Error logging error message: {e}")

    def get_log_files(self) -> tuple[str, str]:
        """
        Get the paths to the current log files.

        Returns:
            tuple[str, str]: Paths to the prompts and responses log files
        """
        return self.prompts_file, self.responses_file
