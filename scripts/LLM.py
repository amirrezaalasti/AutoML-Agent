from groq import Groq


class LLM:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        pass

    def generate(self, prompt, model):
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
