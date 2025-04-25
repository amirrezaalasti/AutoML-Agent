from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

class LLMClient:
    def __init__(self, api_key, model_name, temperature=0):
        self.client = ChatGroq(
            temperature=temperature, groq_api_key=api_key, model_name=model_name
        )

    def generate(self, prompt):

        # Escape curly braces in the context TODO: the prompt should be escaped in the template
        escaped_context = prompt.replace("{", "{{").replace("}", "}}")

        # Create a ChatPromptTemplate with only a system message
        model_prompt = ChatPromptTemplate.from_messages([
            ("system", escaped_context)
        ])

        # Invoke the prompt without any additional input
        prompt_value = model_prompt.invoke({})

        # Pass the formatted prompt to the model
        response = self.client.invoke(prompt_value)
        return response.content

