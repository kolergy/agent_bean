import torch
import ollama
from typing import List
from agent_bean.system_info import SystemInfo

class OllamaEmbeddings:
    """This class wraps the Ollama tokenizer to be used like embeddings"""
    def __init__(self, tokenizer: ollama.OllamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        """Return the token IDs for the text"""
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Return the text for the token IDs"""
        return self.tokenizer.decode(tokens)

    def free(self) -> None:
        """Free the memory used by the tokenizer"""
        self.tokenizer = None

class OllamaModel:
    """This class wraps the Ollama model for generating predictions"""
    def __init__(self, setup: dict, system_info: SystemInfo, model_name: str) -> None:
        self.setup = setup
        self.system_info = system_info
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.instantiate_model()

    def instantiate_model(self) -> None:
        """Instantiate the Ollama model and tokenizer"""
        model_id = self.setup['models_list'][self.model_name]['model_id']
        self.model = ollama.OllamaModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = ollama.OllamaTokenizer.from_pretrained(model_id)

    def predict(self, prompt: str) -> str:
        """Generate text based on the prompt"""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        output_ids = self.model.generate(input_ids, max_length=512)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def free(self) -> None:
        """Free the memory used by the model and tokenizer"""
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Ensure resources are freed when the object is deleted"""
        self.free()
