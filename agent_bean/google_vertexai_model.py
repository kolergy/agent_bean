from typing import List, Dict
from agent_bean.system_info import SystemInfo
from vertexai.preview.generative_models import GenerativeModel, Part

class VertexAIEmbeddings:
    """This class wraps the Google Vertex AI embeddings and provides an interface to access them"""

    def __init__(self, project_id: str, location: str, model_id: str) -> None:
        self.model = GenerativeModel(project=project_id, location=location, model=model_id)
        print(f"VertexAI Embedding initialized with model_id: {model_id}")

    def encode(self, text: str) -> List[int]:
        """Return the embeddings for the query"""
        print(f"VertexAI encode() called with text: {text}")
        # Vertex AI does not directly provide embeddings in this manner, so this is a placeholder
        return []

    def decode(self, tokens: List[int]) -> str:
        """Return the text for the tokens"""
        # Placeholder for decode functionality, if applicable
        return "Decoding not supported"

    def free(self) -> None:
        """Free the memory used by the embeddings"""
        self.model = None

class VertexAIModel:
    """This class wraps the Google Vertex AI API calls"""

    def __init__(self, project_id: str, location: str, model_id: str) -> None:
        self.model = GenerativeModel(project=project_id, location=location, model=model_id)

    def predict(self, prompt: str) -> List[str]:
        """Predict the next token based on the prompt"""
        response = self.model.generate(prompt=[Part(text=prompt)], num_completions=1, max_tokens=50)
        predictions = [completion.text for completion in response.completions[0].parts]
        return predictions

    def __del__(self):
        if self is not None:
            if hasattr(self, 'model'): del self.model
            self.free()

    def free(self) -> None:
        """Free the memory used by the model"""
        if hasattr(self, 'model'): self.model = None
