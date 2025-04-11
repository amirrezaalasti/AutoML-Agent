class LLMClient:
    """Base class for LLM interactions (implement with actual LLM API)"""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError