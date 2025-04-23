from scripts.LLMClient import LLMClient
from typing import Any
from scripts.utils import describe_dataset, log_message
import re


class AutoMLAgent:
    def __init__(self, dataset: Any, llm_client: LLMClient):
        self.dataset = dataset
        self.llm = llm_client
        self.train_function = None
        self.config_space = None
        self.scenario = None
        self.errors = []

    def _extract_function(self, code: str, function_name: str) -> str:
        """Extract the function code from the generated code"""
        pattern = r"```python\n(.*?)```"
        match = re.search(
            pattern, code, re.DOTALL
        )  # Use re.DOTALL to allow dot to match newlines

    def _extract_config_space(self, code: str) -> str:
        """Extract the ConfigurationSpace definition"""
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, code, re.DOTALL)

        return match.group(1) if match else ""

    def _extract_scenario(self, code: str) -> str:
        """Extract the scenario definition"""
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
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

        # Execute the generated code
        self.run_generated_code()

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

    def _inform_errors_to_llm(self, original_promp, code: str) -> str:
        """Inform the LLM about the errors and ask for a fix"""
        with open("templates/fix_prompt.txt", "r") as file:
            template = file.read()
        prompt = template.format(
            original_prompt=original_promp,
            code=code,
            errors="\n".join(self.errors),
        )
        return self.llm.generate(prompt)

    def run_generated_code(self):
        while True:
            try:
                namespace = {}
                # print(f"Executing code:\n{self.config_space}")
                exec(self.config_space, namespace)
                if "get_configspace" in namespace:
                    configuration = namespace["get_configspace"]()
                    print(f"Configuration space:\n{configuration}")
                    log_message(f"Generated configuration space:\n{configuration}")
                    break
            except Exception as e:
                print("****Error occurred during execution:", e)
                error_message = str(e)
                self.errors.append(error_message)
                log_message(f"Error occurred: {error_message}")
                config_prompt = self._create_config_prompt()
                fixed_code = self._inform_errors_to_llm(
                    config_prompt, self.config_space
                )
                self.config_space = self._extract_config_space(fixed_code)
                continue
        self.errors = []
        while True:
            try:
                exec(self.scenario, namespace)
                if "generate_scenario" in namespace:
                    scenario = namespace["generate_scenario"](configuration)
                    log_message(f"Generated scenario: {scenario}")
                    break
            except Exception as e:
                print("****Error occurred during execution:", e)
                error_message = str(e)
                self.errors.append(error_message)
                log_message(f"Error occurred: {error_message}")
                scenario_prompt = self._create_scenario_prompt()
                fixed_code = self._inform_errors_to_llm(scenario_prompt, self.scenario)
                self.scenario = self._extract_scenario(fixed_code)
                continue
        self.errors = []
        # while True:
            # try:
            #     exec(self.train_function, namespace)
            #     if "train" in namespace:
            #         train_function = namespace["train"]
            #         train_function(scenario)
            #         log_message("Training function executed successfully.")
            
            # except Exception as e:
            #     print("****Error occurred during execution:", e)
            #     error_message = str(e)
            #     self.errors.append(error_message)
            #     log_message(f"Error occurred: {error_message}")
            #     train_prompt = self._create_train_prompt()
            #     fixed_code = self._inform_errors_to_llm(
            #         train_prompt, self.train_function
            #     )
            #     self.train_function = self._extract_function(fixed_code, "train")
            #     continue
            # break