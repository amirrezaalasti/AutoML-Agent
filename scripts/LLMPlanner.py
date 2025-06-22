"""
LLM Planner module for AutoML Agent.

This module provides functionality to generate structured instructions for AutoML tasks
using Large Language Models (LLMs) with Google's Gemini API.
"""

import json
import os
from pathlib import Path
from typing import Optional

import instructor
from google import genai
from pydantic import BaseModel

from config.api_keys import GOOGLE_API_KEY
from scripts.Logger import Logger


class InstructorInfo(BaseModel):
    """Structured response model for LLM instructions."""

    recommended_configuration: str
    scenario_plan: str
    train_function_plan: str


class LLMPlannerError(Exception):
    """Custom exception for LLM Planner errors."""

    pass


class LLMPlanner:
    """Planner class that uses LLM to generate structured instructions for AutoML tasks."""

    # Constants
    DEFAULT_MODEL = "gemini-2.0-flash"
    LOGS_DIR = "./logs/instructor"
    SMAC_DOCS_PATH = "collected_docs/smac_docs.json"
    TEMPLATES_DIR = "templates"
    PLANNER_TEMPLATE = "planner_prompt.txt"

    def __init__(
        self,
        dataset_name: str,
        dataset_description: str,
        dataset_type: str,
        task_type: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLMPlanner with dataset details.

        Args:
            dataset_name: Name of the dataset
            dataset_description: Description of the dataset
            dataset_type: Type of the dataset (e.g., classification, regression)
            task_type: Type of task to perform (optional)

        Raises:
            LLMPlannerError: If initialization fails
        """
        try:
            self.dataset_name = dataset_name
            self.dataset_description = dataset_description
            self.dataset_type = dataset_type
            self.task_type = task_type or "classification"  # Default task type

            # Initialize logger
            self.logger = Logger(
                model_name=self.DEFAULT_MODEL,
                dataset_name=self.dataset_name,
                base_log_dir=self.LOGS_DIR,
            )

            # Load prompt template
            self._load_prompt_template()

        except Exception as e:
            raise LLMPlannerError(f"Failed to initialize LLMPlanner: {e}")

    def _load_prompt_template(self) -> None:
        """
        Load the planner prompt template from file.

        Raises:
            LLMPlannerError: If template file cannot be loaded
        """
        try:
            template_path = Path(self.TEMPLATES_DIR) / self.PLANNER_TEMPLATE

            if not template_path.exists():
                raise LLMPlannerError(f"Planner template not found: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                self.planner_template = f.read()

        except Exception as e:
            raise LLMPlannerError(f"Failed to load prompt template: {e}")

    def _get_smac_documentation(self) -> str:
        """
        Get SMAC documentation content if available.

        Returns:
            SMAC documentation content or empty string if not available
        """
        smac_docs_path = Path(self.SMAC_DOCS_PATH)

        if not smac_docs_path.exists():
            return ""

        try:
            with open(smac_docs_path, "r", encoding="utf-8") as f:
                smac_docs = json.load(f)

            # Extract documentation content
            doc_content = "\n\n".join([doc["content"] for doc in smac_docs])
            return doc_content

        except (json.JSONDecodeError, IOError) as e:
            self.logger.log_warning(f"Error reading SMAC documentation: {e}")
            return ""

    def generate_instructions(self, config_space_suggested_parameters: Optional[str] = None) -> InstructorInfo:
        """
        Generate structured instructions for the dataset using the Gemini LLM.

        Args:
            config_space_suggested_parameters: Optional parameters from OpenML for configuration space

        Returns:
            InstructorInfo: Structured response containing the generated instructions

        Raises:
            LLMPlannerError: If instruction generation fails
        """
        try:
            # Get SMAC documentation
            smac_documentation = self._get_smac_documentation()

            # Create the complete instruction using the planner template
            instruction = self.planner_template.format(
                dataset_name=self.dataset_name,
                dataset_description=self.dataset_description,
                dataset_type=self.dataset_type,
                task_type=self.task_type,
                smac_documentation=smac_documentation,
            )

            # Initialize LLM client
            google_client = genai.Client(api_key=GOOGLE_API_KEY)
            client = instructor.from_genai(google_client, model=f"models/{self.DEFAULT_MODEL}")

            # Generate response
            response = client.chat.completions.create(
                response_model=InstructorInfo,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": "generate the instructions"},
                ],
            )

            # Log the interaction
            self.logger.log_prompt(prompt=instruction, metadata={"dataset_name": self.dataset_name})
            self.logger.log_response(response, metadata={"dataset_name": self.dataset_name})

            return response

        except Exception as e:
            raise LLMPlannerError(f"Failed to generate instructions: {e}")
