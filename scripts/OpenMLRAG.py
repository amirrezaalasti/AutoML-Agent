import openml
from configs.api_keys import OPENML_API_KEY


class OpenMLRAG:
    def __init__(self):
        openml.config.apikey = OPENML_API_KEY

    def run(self):
        pass
