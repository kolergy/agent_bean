import google.generativeai as genai
from typing import List

class GeminiEmbeddings:
    """This class wraps the Google Gemini embeddings and provides an interface to access them"""

    def __init__(self, model_name: str) -> None:
        self.model = genai.GenerativeModel(model_name=model_name)
        print(f"Gemini Embedding initialized with model_name: {model_name}")

    def encode(self, text: str) -> List[int]:
        """Return the embeddings for the query"""
        print(f"Gemini encode() called with text: {text}")
        # Placeholder: Gemini does not directly provide embeddings in this manner
        return []

    def decode(self, tokens: List[int]) -> str:
        """Return the text for the tokens"""
        # Placeholder for decode functionality, if applicable
        return "Decoding not supported"

    def free(self) -> None:
        """Free the memory used by the embeddings"""
        self.model = None

class GeminiModel:
    """This class wraps the Google Gemini API calls"""

    def __init__(self, model_name: str) -> None:
        self.model = genai.GenerativeModel(model_name=model_name)

    def predict(self, prompt: str) -> List[str]:
        """Predict the next token based on the prompt"""
        response = self.model.generate(prompt=prompt, num_completions=1, max_tokens=50)
        predictions = [completion.text for completion in response.completions]
        return predictions

    def __del__(self):
        if self is not None:
            if hasattr(self, 'model'): del self.model
            self.free()

    def free(self) -> None:
        """Free the memory used by the model"""
        if hasattr(self, 'model'): self.model = None

    @staticmethod
    def keyify_model_id(model_id):
        """Clean the model_id into a string that can be used as a key"""
        return str(model_id).replace('/', '_-_')

    @staticmethod
    def de_keyify_model_id(cleaned_model_id):
        """Reverse the cleaning process to get the original model_id"""
        return cleaned_model_id.replace('_-_', '/')
