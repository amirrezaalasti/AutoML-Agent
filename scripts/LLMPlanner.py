"""
LLM Planner module for AutoML Agent.

This module provides functionality to generate structured instructions for AutoML tasks
using Large Language Models (LLMs) with Google's Gemini API.
"""

import json
import os
from pathlib import Path
from typing import Optional

from google import genai
from pydantic import BaseModel
from openai import OpenAI

from config.api_keys import GOOGLE_API_KEY, LOCAL_LLAMA_API_KEY
from scripts.Logger import Logger


class InstructorInfo(BaseModel):
    """Structured response model for LLM instructions."""

    configuration_plan: str
    scenario_plan: str
    train_function_plan: str
    suggested_facade: str


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
    PLANNER_TEMPLATE_TABULAR = "planner_prompt_tabular.txt"
    PLANNER_TEMPLATE_IMAGE = "planner_prompt_image.txt"

    def __init__(
        self,
        dataset_name: str,
        dataset_description: str,
        dataset_type: str,
        task_type: Optional[str] = None,
        model_name: str = None,
        base_url: str = None,
        api_key: str = None,
    ) -> None:
        """
        Initialize the LLMPlanner with dataset details.

        Args:
            dataset_name: Name of the dataset
            dataset_description: Description of the dataset
            dataset_type: Type of the dataset
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
                model_name=model_name,
                dataset_name=self.dataset_name,
                base_log_dir=self.LOGS_DIR,
            )
            self.base_url = base_url
            if self.base_url:
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
                self.model_name = model_name
            else:
                # Initialize Google client
                self.client = genai.Client(
                    api_key=api_key,
                )

            # Load appropriate prompt template based on dataset type
            self._load_prompt_template()

        except Exception as e:
            raise LLMPlannerError(f"Failed to initialize LLMPlanner: {e}")

    def _get_template_name(self) -> str:
        """
        Determine the appropriate template name based on dataset type.

        Returns:
            Template filename for the dataset type
        """
        # Check if dataset type suggests image data
        if any(
            keyword in self.dataset_type.lower()
            for keyword in ["image", "vision", "cnn", "conv"]
        ):
            return self.PLANNER_TEMPLATE_IMAGE

        # Check if dataset description suggests image data
        if any(
            keyword in self.dataset_description.lower()
            for keyword in ["image", "pixel", "vision", "photo", "picture"]
        ):
            return self.PLANNER_TEMPLATE_IMAGE

        # Default to tabular template
        return self.PLANNER_TEMPLATE_TABULAR

    def _load_prompt_template(self) -> None:
        """
        Load the appropriate planner prompt template from file based on dataset type.

        Raises:
            LLMPlannerError: If template file cannot be loaded
        """
        try:
            template_name = self._get_template_name()
            template_path = Path(self.TEMPLATES_DIR) / template_name

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

    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from LLM output.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            LLMPlannerError: If JSON parsing fails
        """
        try:
            # Try to find JSON content between ```json and ``` markers
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    json_content = response_text[start:end].strip()
                else:
                    json_content = response_text[start:].strip()
            else:
                # Try to find JSON object directly
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    json_content = response_text[start:end]
                else:
                    json_content = response_text

            return json.loads(json_content)

        except json.JSONDecodeError as e:
            raise LLMPlannerError(f"Failed to parse JSON response: {e}")

    def generate_instructions(
        self,
        config_space_suggested_parameters: Optional[str] = None,
        improvement_context: Optional[str] = None,
    ) -> InstructorInfo:
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

            # Create the base instruction using the planner template
            instruction = self.planner_template.format(
                dataset_name=self.dataset_name,
                dataset_description=self.dataset_description,
                dataset_type=self.dataset_type,
                task_type=self.task_type,
                config_space_suggested_parameters=config_space_suggested_parameters
                or "No OpenML meta-learning insights available for this dataset.",
            )
            # Append improvement context if provided (for iterative planning)
            if improvement_context:
                instruction += "\n\n### Iterative Improvement Context\n" + str(
                    improvement_context
                )
            if self.base_url:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": instruction}],
                )
                response_text = response.choices[0].message.content
            else:
                # Generate response using Google Gemini
                response = self.client.models.generate_content(
                    model=f"models/{self.DEFAULT_MODEL}",
                    contents=[{"parts": [{"text": instruction}]}],
                )
                # Extract response text
                response_text = response.candidates[0].content.parts[0].text

            # Parse JSON response
            parsed_response = self._parse_json_response(response_text)

            # Validate required keys
            required_keys = [
                "configuration_plan",
                "scenario_plan",
                "train_function_plan",
                "suggested_facade",
            ]
            for key in required_keys:
                if key not in parsed_response:
                    raise LLMPlannerError(f"Missing required key in response: {key}")

            # Create InstructorInfo object
            instructor_info = InstructorInfo(
                configuration_plan=parsed_response["configuration_plan"],
                scenario_plan=parsed_response["scenario_plan"],
                train_function_plan=parsed_response["train_function_plan"],
                suggested_facade=parsed_response["suggested_facade"],
            )

            # Log the interaction
            self.logger.log_prompt(
                prompt=instruction, metadata={"dataset_name": self.dataset_name}
            )
            self.logger.log_response(
                response=response_text, metadata={"dataset_name": self.dataset_name}
            )

            return instructor_info

        except Exception as e:
            raise LLMPlannerError(f"Failed to generate instructions: {e}")
