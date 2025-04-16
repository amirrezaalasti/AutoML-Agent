from scripts.LLMClient import LLMClient
from typing import Any
from scripts.utils import describe_dataset, log_message
from scripts.CodeExecutor import CodeExecutor


class AutoMLAgent:
    def __init__(self, dataset: Any, llm_client: LLMClient):
        self.dataset = dataset
        self.llm = llm_client
        self.train_function = None
        self.config_space = None
        self.scenario = None

    def _extract_function(self, code: str, function_name: str) -> str:
        """Extract the function code from the generated code"""
        import re

        pattern = rf"(def {function_name}\(.*?\):(?:\n    .*)+)"
        match = re.search(pattern, code)
        return match.group(1) if match else ""

    def _extract_config_space(self, code: str) -> str:
        """Extract the ConfigurationSpace definition"""
        import re

        pattern = r"(cs\s*=\s*ConfigurationSpace\(\)(?:.|\n)*?)\n\n"
        match = re.search(pattern, code)
        return match.group(1) if match else ""

    def _extract_scenario(self, code: str) -> str:
        """Extract the Scenario definition"""
        import re

        pattern = r"(scenario\s*=\s*Scenario\((?:.|\n)*?\))"
        match = re.search(pattern, code)
        return match.group(1) if match else ""

    def generate_components(self):
        """Generate all necessary components through LLM prompts"""
        # Generate training function
        train_prompt = self._create_train_prompt()
        train_code = self.llm.generate(train_prompt)
        log_message(f"Generated training function code:\n{train_code}")
        self.train_function_code = train_code
        self.train_function = self._extract_function(train_code, "train")

        # Generate configuration space
        config_prompt = self._create_config_prompt()
        config_code = self.llm.generate(config_prompt)
        log_message(f"Generated configuration space code:\n{config_code}")
        self.config_code = config_code
        self.config_space = self._extract_config_space(config_code)

        # Generate scenario
        scenario_prompt = self._create_scenario_prompt()
        scenario_code = self.llm.generate(scenario_prompt)
        log_message(f"Generated scenario code:\n{scenario_code}")
        self.scenario_code = scenario_code
        self.scenario = self._extract_scenario(scenario_code)

    def _create_train_prompt(self) -> str:
        with open("templates/train_prompt.txt", "r") as file:
            template = file.read()
        return template

    def _create_config_prompt(self) -> str:
        with open("templates/config_prompt.txt", "r") as file:
            template = file.read()
        return template.format(
            dataset=describe_dataset(self.dataset),
        )

    def _create_scenario_prompt(self) -> str:
        with open("templates/scenario_prompt.txt", "r") as file:
            template = file.read()
        return template.format(
            dataset=describe_dataset(self.dataset),
        )

    def run_smac(self):
        pass
