from groq import Groq


class LLMClient:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        pass

    def generate(self, prompt, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return completion.choices[0].message.content
