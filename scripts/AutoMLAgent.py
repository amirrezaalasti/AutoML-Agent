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
        return f"""Create a Python training function with these requirements:
        - Dataset: {describe_dataset(self.dataset)}
        - Use appropriate validation strategy
        - Must return validation loss
        - Signature: def train(cfg: Configuration, seed: int) -> float
        - Should handle hyperparameters from config
        - Include necessary imports
        Example:
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import SGDClassifier
        
        def train(cfg, seed):
            model = SGDClassifier(
                alpha=cfg['alpha'],
                learning_rate=cfg['learning_rate'],
                max_iter=cfg['max_iter']
            )
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            return 1 - np.mean(scores)
        """

    def _create_config_prompt(self) -> str:
        return """Create ConfigSpace for:
        - ML algorithm used in train function
        - Include relevant hyperparameters
        - Specify proper ranges/types
        Example:
        from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
        
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter('alpha', 1e-5, 1e-1, log=True))
        """

    def _create_scenario_prompt(self) -> str:
        return """Create a scenario for SMAC:
        - Use the generated training function
        - Include necessary imports
        Example:
        from smac.scenario.scenario import Scenario
        scenario = Scenario({
            'runcount-limit': 100,
            'cs': cs,
            'deterministic': True,
            'output-dir': './output',
            'run_obj': 'quality'
        })
        """

    def run_smac(self):
        pass
