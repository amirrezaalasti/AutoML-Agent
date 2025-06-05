from pydantic import BaseModel
import instructor
from configs.api_keys import GOOGLE_API_KEY  # Your API key
from google import genai
from scripts.Logger import Logger


class InstructorInfo(BaseModel):
    dataset_name: str
    dataset_tag: str
    dataset_instructions: list[str]
    openml_url: str


class LLMPlanner:
    def __init__(self, dataset_name: str, dataset_description: str, dataset_type: str):
        self.dataset_name = dataset_name
        self.dataset_description = dataset_description
        self.dataset_type = dataset_type
        self.instruction = (
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
            "   - Provide the direct OpenML URL to the dataset.\n\n"
            "Return all the above in a structured JSON format matching the following fields:\n"
            "- dataset_name (str)\n"
            "- dataset_tag (str)\n"
            "- dataset_instructions (list of str)\n"
        )

        self.logger = Logger(
            model_name="gemini-2.0-flash",
            dataset_name=self.dataset_name,
            base_log_dir="./logs/instructor",
        )

    def generate_instructions(self):
        google_client = genai.Client(api_key=GOOGLE_API_KEY)
        client = instructor.from_genai(google_client, model="models/gemini-2.0-flash")
        response = client.chat.completions.create(
            response_model=InstructorInfo,
            messages=[
                {
                    "role": "user",
                    "content": self.instruction,
                }
            ],
        )

        self.logger.log_prompt(prompt=self.instruction, metadata={"dataset_name": self.dataset_name})
        self.logger.log_response(response, metadata={"dataset_name": self.dataset_name})
        return response
