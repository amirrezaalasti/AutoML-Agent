from sklearn.datasets import load_iris
from scripts.AutoMLAgent import AutoMLAgent
from ConfigSpace import ConfigurationSpace

class DummyLLM:
    def generate(self, prompt):
        # Simulate LLM response for training function
        if "training function" in prompt:
            return ""


X, y = load_iris(return_X_y=True)

agent = AutoMLAgent(dataset={'X': X, 'y': y}, llm_client=DummyLLM())
agent.generate_components()
incumbent = agent.run_smac()
print(f"Best configuration: {incumbent}")