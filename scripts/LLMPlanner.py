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

    # Template file names
    TEMPLATE_FILES = {
        "base": "planner_base_prompt.txt",
        "config": "planner_config_prompt.txt",
        "scenario": "planner_scenario_prompt.txt",
        "train": "planner_train_prompt.txt",
    }

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

            # Load prompt templates
            self._load_prompt_templates()

        except Exception as e:
            raise LLMPlannerError(f"Failed to initialize LLMPlanner: {e}")

    def _load_prompt_templates(self) -> None:
        """
        Load prompt templates from template files.

        Raises:
            LLMPlannerError: If template files cannot be loaded
        """
        try:
            templates_dir = Path(self.TEMPLATES_DIR)

            # Load base instruction template
            base_template_path = templates_dir / self.TEMPLATE_FILES["base"]
            if not base_template_path.exists():
                raise LLMPlannerError(f"Base template not found: {base_template_path}")

            with open(base_template_path, "r", encoding="utf-8") as f:
                self.base_template = f.read()

            # Load configuration instruction template
            config_template_path = templates_dir / self.TEMPLATE_FILES["config"]
            if not config_template_path.exists():
                raise LLMPlannerError(f"Config template not found: {config_template_path}")

            with open(config_template_path, "r", encoding="utf-8") as f:
                self.config_template = f.read()

            # Load scenario instruction template
            scenario_template_path = templates_dir / self.TEMPLATE_FILES["scenario"]
            if not scenario_template_path.exists():
                raise LLMPlannerError(f"Scenario template not found: {scenario_template_path}")

            with open(scenario_template_path, "r", encoding="utf-8") as f:
                self.scenario_template = f.read()

            # Load train function instruction template
            train_template_path = templates_dir / self.TEMPLATE_FILES["train"]
            if not train_template_path.exists():
                raise LLMPlannerError(f"Train template not found: {train_template_path}")

            with open(train_template_path, "r", encoding="utf-8") as f:
                self.train_template = f.read()

        except Exception as e:
            raise LLMPlannerError(f"Failed to load prompt templates: {e}")

    def _create_base_instruction(self) -> str:
        """
        Create the base instruction for the LLM using template.

        Returns:
            Formatted base instruction string
        """
        return self.base_template.format(
            dataset_name=self.dataset_name,
            dataset_description=self.dataset_description,
            dataset_type=self.dataset_type,
            task_type=self.task_type,
        )

    def _create_configuration_instruction(self, config_space_suggested_parameters: str) -> str:
        """
        Create the configuration instruction using template.

        Args:
            config_space_suggested_parameters: Suggested parameters for configuration

        Returns:
            Formatted configuration instruction string
        """
        return self.config_template.format(config_space_suggested_parameters=config_space_suggested_parameters)

    def _create_scenario_instruction(self) -> Optional[str]:
        """
        Create the scenario instruction based on SMAC documentation if available.

        Returns:
            Formatted scenario instruction string or None if SMAC docs not available
        """
        smac_docs_path = Path(self.SMAC_DOCS_PATH)

        if not smac_docs_path.exists():
            return None

        try:
            with open(smac_docs_path, "r", encoding="utf-8") as f:
                smac_docs = json.load(f)

            # Extract documentation content
            doc_content = "\n\n".join([doc["content"] for doc in smac_docs])

            return self.scenario_template.format(smac_documentation=doc_content)

        except (json.JSONDecodeError, IOError) as e:
            raise LLMPlannerError(f"Error reading SMAC documentation: {e}")

    def _create_train_function_instruction(self) -> str:
        """
        Create the train function instruction using template.

        Returns:
            Train function instruction string
        """
        return self.train_template

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
            # Build complete instruction
            instruction = self._create_base_instruction()

            if config_space_suggested_parameters:
                instruction += self._create_configuration_instruction(config_space_suggested_parameters)

            scenario_instruction = self._create_scenario_instruction()
            if scenario_instruction:
                instruction += scenario_instruction

            instruction += self._create_train_function_instruction()

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
