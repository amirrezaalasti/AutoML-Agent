# from sklearn.datasets import load_iris
# from scripts.AutoMLAgent import AutoMLAgent
# from ConfigSpace import ConfigurationSpace
from configs.api_keys import GROQ_API_KEY
from scripts.LLM import LLM

llm = LLM(api_key=GROQ_API_KEY)
response = llm.generate(
    prompt="Generate a configuration space for a random forest classifier with max_depth between 1 and 10.",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)
print(response)