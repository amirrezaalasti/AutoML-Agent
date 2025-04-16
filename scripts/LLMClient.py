from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage


class LLMClient:
    def __init__(self, api_key, model_name, temparature=0):
        self.client = ChatGroq(
            temperature=temparature, groq_api_key=api_key, model_name=model_name
        )

    def generate(self, prompt):
        context = (
            "You are an AutoML expert creating components for SMAC optimization. "
            "Components must be Python code that strictly follows these requirements:\n"
            "1. Training function with proper hyperparameter handling\n"
            "2. ConfigurationSpace with proper constraints\n"
            "3. SMAC Scenario configuration\n"
            "All components must be compatible and use consistent hyperparameter names."
        )
        
        model_prompt = ChatPromptTemplate.from_messages(
            [("system", context), ("human", prompt)]
        )

        response = self.client.invoke(input=prompt)
        return response.content
