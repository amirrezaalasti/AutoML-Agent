from typing import Optional
from pydantic import BaseModel
import instructor
from google import genai
import os
import json
from config.api_keys import GOOGLE_API_KEY
from scripts.Logger import Logger


class InstructorInfo(BaseModel):
    """Structured response model for LLM instructions."""

    dataset_name: str
    dataset_tag: str
    recommended_configuration: str
    scenario_plan: str
    train_function_plan: str


class LLMPlanner:
    """Planner class that uses LLM to generate structured instructions for AutoML tasks."""

    # Constants
    DEFAULT_MODEL = "gemini-2.0-flash"
    LOGS_DIR = "./logs/instructor"
    SMAC_DOCS_PATH = "collected_docs/smac_docs.json"

    def __init__(
        self,
        dataset_name: str,
        dataset_description: str,
        dataset_type: str,
        task_type: str | None = None,
    ) -> None:
        """
        Initialize the LLMPlanner with dataset details.

        Args:
            dataset_name: Name of the dataset
            dataset_description: Description of the dataset
            dataset_type: Type of the dataset (e.g., classification, regression)
        """
        self.dataset_name = dataset_name
        self.dataset_description = dataset_description
        self.dataset_type = dataset_type
        self.task_type = task_type
        # Initialize instructions
        self.instruction = self._create_base_instruction()
        self.instruction_for_configuration = self._create_configuration_instruction()
        self.scenario_plan_instruction = self._create_scenario_instruction()
        self.train_function_plan_instruction = self._create_train_function_instruction()

        # Initialize logger
        self.logger = Logger(
            model_name=self.DEFAULT_MODEL,
            dataset_name=self.dataset_name,
            base_log_dir=self.LOGS_DIR,
        )

    def _create_base_instruction(self) -> str:
        """Create the base instruction for the LLM."""
        return (
            f"You are a data science expert with deep knowledge of {self.dataset_type} datasets. "
            f"Your task is to generate structured, and comprehensive instructions for effectively working with the dataset '{self.dataset_name}'. "
            f"Dataset description: {self.dataset_description}\n\n"
            "Please perform the following tasks:\n"
            "1. Describe the recommended data preprocessing steps specific to this dataset.\n"
            "2. Outline useful feature engineering strategies relevant to the dataset's type and domain.\n"
            "3. Mention any common challenges or important considerations when working with this dataset type.\n"
            "4. If this dataset exists on OpenML:\n"
            "   - Provide the exact dataset name as listed on OpenML.\n"
            "   - Include the dataset's tag(s) on OpenML.\n"
            "Return all the above in a structured JSON format matching the following fields:\n"
            "- dataset_name (str)\n"
            "- dataset_tag (str)\n"
            "- recommended_configuration (str)\n"
            "- scenario_plan (str)\n"
            "- train_function_plan (str)\n"
            "- you have to do the task type: {self.task_type}\n"
            "- do not suggest error handling code, the agent should not handle errors, it should be handled by the user."
            "- your instructions should be a step by step guide for the agent to follow."
            "- if a dataset and a task needs to be done on GPU, you should suggest to use GPU."
        )

    def _create_configuration_instruction(self) -> str:
        """Create the configuration instruction template."""
        return """
            Below, I am providing a set of parameters that were used on the top-performing models evaluated on the dataset.
            Please generate an instruction for creating a configuration for the dataset based on the following parameters:
            {config_space_suggested_parameters}
            * This is just a Suggestion, you can use it as a reference,
            but you can also ignore it because the task type can be different from your goal and generate your own configuration space.
            * for recommended_configuration you should suggest the best parameters to set in Configuration Space.
            """

    def _create_scenario_instruction(self) -> Optional[str]:
        """Create the scenario instruction based on SMAC documentation if available."""
        if not os.path.exists(self.SMAC_DOCS_PATH):
            return None

        try:
            with open(self.SMAC_DOCS_PATH, "r") as f:
                smac_docs = json.load(f)

            doc_context = """
            Based on the following SMAC documentation, analyze the dataset characteristics and choose appropriate:
            1. Facade type (e.g., MultiFidelityFacade for multi-fidelity optimization)
            2. Budget settings (min_budget and max_budget)
            3. Number of workers (n_workers)
            4. Other relevant scenario parameters

            SMAC Documentation:
            """
            doc_context += "\n\n".join([doc["content"] for doc in smac_docs])

            doc_context += """
            Please analyze the dataset and documentation to determine:
            1. Should multi-fidelity optimization be used? (Consider dataset size and training time)
            2. What budget range is appropriate? (Consider training epochs or data subsets)
            3. How many workers should be used? (Consider available resources)
            4. Are there any special considerations for this dataset type?
            5. Do not suggest name and output directory for the scenario, it will be generated by the agent.
            6. Do not suggest any code for the scenario, it will be generated by the agent.

            Then generate a scenario configuration that best suits this dataset.
            """
            return doc_context
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading SMAC documentation: {e}")
            return None

    def _create_train_function_instruction(self) -> str:
        """Create the train function instruction."""
        return """
        Based on the previous instructions and knowledge of the dataset, generate a plan for the train function.
        * Do not suggest any code for the train function, it will be generated by the agent.
        """

    def generate_instructions(self, config_space_suggested_parameters: Optional[str] = None) -> InstructorInfo:
        """
        Generate structured instructions for the dataset using the Gemini LLM.

        Args:
            config_space_suggested_parameters: Optional parameters from OpenML for configuration space

        Returns:
            InstructorInfo: Structured response containing the generated instructions
        """
        # Build complete instruction
        instruction = self.instruction
        if config_space_suggested_parameters:
            instruction += self.instruction_for_configuration.format(config_space_suggested_parameters=config_space_suggested_parameters)
        if self.scenario_plan_instruction:
            instruction += self.scenario_plan_instruction
        if self.train_function_plan_instruction:
            instruction += self.train_function_plan_instruction

        # Initialize LLM client
        google_client = genai.Client(api_key=GOOGLE_API_KEY)
        client = instructor.from_genai(google_client, model=f"models/{self.DEFAULT_MODEL}")

        # Generate response
        response = client.chat.completions.create(
            response_model=InstructorInfo,
            messages=[{"role": "user", "content": instruction}],
        )

        # Log the interaction
        self.logger.log_prompt(prompt=instruction, metadata={"dataset_name": self.dataset_name})
        self.logger.log_response(response, metadata={"dataset_name": self.dataset_name})

        return response
