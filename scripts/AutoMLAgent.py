from scripts.LLMClient import LLMClient
from typing import Any
from scripts.utils import describe_dataset, log_message, save_code
import re


def extract_code_block(code: str) -> str:
    """Extract Python code between markdown code fences"""
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1) if match else code


class AutoMLAgent:
    MAX_RETRIES_PROMPT = 5
    MAX_RETRIES_ABORT = 20

    def __init__(
        self, dataset: Any, llm_client: LLMClient, dataset_type: str, ui_agent: Any
    ):
        self.dataset = dataset
        self.llm = llm_client
        self.dataset_type = dataset_type
        self.dataset_description = describe_dataset(dataset, dataset_type)

        self.config_code = None
        self.scenario_code = None
        self.train_function_code = None

        self.config_space = None
        self.scenario = None
        self.train_function = None

        self.error_counters = {"config": 0, "scenario": 0, "train_function": 0}
        self.error_history = {"config": [], "scenario": [], "train_function": []}
        self.namespace = {}

        # store original prompts for retries
        self.prompts = {}
        self.ui_agent = ui_agent

    def generate_components(self):
        # Configuration Space
        self.prompts["config"] = self._create_config_prompt()
        self.config_code = self.llm.generate(self.prompts["config"])
        self._run_generated("config")
        self.ui_agent.subheader("Generated Configuration Space Code")
        self.ui_agent.code(self.config_space, language="python")

        # Scenario
        self.prompts["scenario"] = self._create_scenario_prompt()
        self.scenario_code = self.llm.generate(self.prompts["scenario"])
        self._run_generated("scenario")
        self.ui_agent.subheader("Generated Scenario Code")
        self.ui_agent.code(self.scenario, language="python")

        # Training Function
        self.prompts["train_function"] = self._create_train_prompt()
        self.train_function_code = self.llm.generate(self.prompts["train_function"])
        log_message(f"Generated training function code:\n{self.train_function_code}")
        self._run_generated("train_function")
        self.ui_agent.subheader("Generated Training Function Code")
        self.ui_agent.code(self.train_function, language="python")

        print("self.config_space", self.config_space)
        print("self.scenario", self.scenario)
        print("self.train_function", self.train_function)
        # Save outputs
        save_code(self.config_space, "generated_config_space.py")
        save_code(self.scenario, "generated_scenario.py")
        save_code(self.train_function, "generated_train_function.py")

        # Return results and last training loss
        return (
            self.config_space,
            self.scenario,
            self.train_function,
            self.last_loss,
            self.prompts,
        )

    def _run_generated(self, component: str):
        """Generic runner for config, scenario, or train_function"""
        retry_count = 0
        while True:
            code_attr = f"{component}_code"
            raw_code = getattr(self, code_attr)
            source = extract_code_block(raw_code)
            log_message(f"Running {component} code:\n{source}")

            try:
                if component == "config":
                    exec(source, self.namespace)
                    cfg = self.namespace["get_configspace"]()
                    self.config_space = source
                    log_message(f"Configuration space generated: {cfg}")

                elif component == "scenario":
                    cfg = self.namespace["get_configspace"]()
                    exec(source, self.namespace)
                    scenario_obj = self.namespace["generate_scenario"](cfg)
                    self.scenario = source
                    log_message(f"Scenario generated: {scenario_obj}")

                elif component == "train_function":
                    cfg = self.namespace["get_configspace"]()
                    exec(source, self.namespace)
                    train_fn = self.namespace["train"]
                    sampled = cfg.sample_configuration()
                    loss = train_fn(cfg=sampled, seed=42, dataset=self.dataset)
                    self.train_function = source
                    self.last_loss = loss
                    log_message(f"Training executed, loss: {loss}")

                return  # success

            except Exception as e:
                retry_count += 1
                self.error_counters[component] += 1
                err = str(e)
                # record error history
                self.error_history[component].append(err)
                log_message(f"Error in {component} (#{retry_count}): {err}")

                # Abort after too many retries
                if self.error_counters[component] > self.MAX_RETRIES_ABORT:
                    log_message(
                        f"Max abort retries reached for {component}. Using last generated code."
                    )
                    if component == "train_function":
                        self.last_loss = getattr(self, "last_loss", None)
                    return

                # Refresh code from original prompt after threshold
                if retry_count == self.MAX_RETRIES_PROMPT:
                    log_message(
                        f"Retry limit reached for {component}. Fetching fresh code from LLM."
                    )
                    fresh = self.llm.generate(self.prompts[component])
                    setattr(self, code_attr, fresh)
                    retry_count = 0
                else:
                    # Ask LLM to fix errors with history of recent issues
                    recent_errors = "\n".join(
                        self.error_history[component][-self.MAX_RETRIES_PROMPT :]
                    )
                    fix_prompt = self._create_fix_prompt(recent_errors, raw_code)
                    fixed = self.llm.generate(fix_prompt)
                    setattr(self, code_attr, fixed)

    def _create_config_prompt(self) -> str:
        with open("templates/config_prompt.txt", "r") as f:
            return f.read().format(dataset_description=self.dataset_description)

    def _create_scenario_prompt(self) -> str:
        with open("templates/scenario_prompt.txt", "r") as f:
            return f.read().format(dataset=self.dataset_description)

    def _create_train_prompt(self) -> str:
        with open("templates/train_prompt.txt", "r") as f:
            return f.read().format(
                dataset_description=self.dataset_description,
                config_space=self.config_space or "",
                scenario=self.scenario or "",
            )

    def _create_fix_prompt(self, errors: str, code: str) -> str:
        """Include recent error history when asking LLM to fix code"""
        with open("templates/fix_prompt.txt", "r") as f:
            return f.read().format(code=code, errors=errors)
