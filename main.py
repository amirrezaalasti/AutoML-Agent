from sklearn.datasets import load_iris
from scripts.AutoMLAgent import AutoMLAgent
from ConfigSpace import ConfigurationSpace
from configs.api_keys import GROQ_API_KEY
from scripts.LLMClient import LLMClient

def generate_response():
    llm = LLMClient(api_key=GROQ_API_KEY)
    response = llm.generate(
        prompt="Generate a configuration space for a random forest classifier with max_depth between 1 and 10.",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    print(response)

X, y = load_iris(return_X_y=True)

# agent.generate_components()
# incumbent = agent.run_smac()
# print(f"Best configuration: {incumbent}")

if __name__ == "__main__":
    llm_client = LLMClient(api_key=GROQ_API_KEY)
    agent = AutoMLAgent(dataset={'X': X, 'y': y}, llm_client=llm_client)
    agent.generate_components()
    # incumbent = agent.run_smac()
    # print(f"Best configuration: {incumbent}")
    pass