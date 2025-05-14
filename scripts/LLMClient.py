from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from google import genai


class LLMClient:
    def __init__(self, api_key, model_name, temperature=0):
        # if model name has gemini in it, use genai
        if "gemini" in model_name:
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name

        else:
            # Use ChatGroq for other models
            self.client = ChatGroq(temperature=temperature, groq_api_key=api_key, model_name=model_name)
            self.model_name = model_name

    def generate(self, prompt):
        # Escape curly braces in the context TODO: the prompt should be escaped in the template
        escaped_context = prompt.replace("{", "{{").replace("}", "}}")

        if "gemini" in self.model_name:
            response = self.client.models.generate_content(model="gemini-2.0-flash", contents=escaped_context)
            return response.text

        # Create a ChatPromptTemplate with only a system message
        model_prompt = ChatPromptTemplate.from_messages([("system", escaped_context)])

        # Invoke the prompt without any additional input
        prompt_value = model_prompt.invoke({})

        # Pass the formatted prompt to the model
        response = self.client.invoke(prompt_value)
        return response.content
