import os
from datetime import datetime
from typing import Optional


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
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.base_log_dir = base_log_dir

        # Create log directory structure
        self.experiment_dir = os.path.join(
            self.base_log_dir,
            f"{self.model_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        self.prompts_file = os.path.join(self.experiment_dir, "prompts.log")
        self.responses_file = os.path.join(self.experiment_dir, "responses.log")

        # Create directory if it doesn't exist
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _write_log(self, message: str, file_path: str, metadata: Optional[dict] = None):
        """
        Write a log message to the specified file with timestamp and optional metadata.

        Args:
            message (str): The message to log
            file_path (str): Path to the log file
            metadata (dict, optional): Additional metadata to include in the log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] "

        if metadata:
            log_entry += f"[Metadata: {metadata}] "

        log_entry += f"{message}\n"
        log_entry += "-" * 80 + "\n"  # Add separator line

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_prompt(self, prompt: str, metadata: Optional[dict] = None):
        """
        Log a prompt message.

        Args:
            prompt (str): The prompt to log
            metadata (dict, optional): Additional metadata about the prompt
        """
        self._write_log(prompt, self.prompts_file, metadata)

    def log_response(self, response: str, metadata: Optional[dict] = None):
        """
        Log a response message.

        Args:
            response (str): The response to log
            metadata (dict, optional): Additional metadata about the response
        """
        self._write_log(response, self.responses_file, metadata)

    def log_error(self, error_message: str, error_type: str = "ERROR"):
        """
        Log an error message to both prompt and response files.

        Args:
            error_message (str): The error message to log
            error_type (str): Type of error (default: "ERROR")
        """
        metadata = {"error_type": error_type}
        error_entry = f"[{error_type}] {error_message}"
        self._write_log(error_entry, self.prompts_file, metadata)
        self._write_log(error_entry, self.responses_file, metadata)

    def get_log_files(self) -> tuple[str, str]:
        """
        Get the paths to the current log files.

        Returns:
            tuple[str, str]: Paths to the prompts and responses log files
        """
        return self.prompts_file, self.responses_file
