
from   typing                         import List, Dict
from   agent_bean.system_info         import SystemInfo

from mistralai.client                 import MistralClient
from mistralai.models.chat_completion import ChatMessage

class MistralEmbeddings:
    """this class warps the mistral embeddings and provides a interface to access them""" 
    
    def __init__(self, Mistral_api_key: str, model_name: str) -> None:
        self.client     = MistralClient(api_key=Mistral_api_key)
        self.model_name = model_name
        self.last_text  = ""
        print(f"Mistral Embeding initialised with model_name: {model_name}")
                
    #def __call__(self, text: str):
    #    """Return the embeddings for the text"""

    def encode(self, text: str) -> List[int]:
        """Return the embeddings for the query"""
        print(f"Mistral encode() called with text: {text}")
        self.last_text  = text
        embeddings_response = self.client.embeddings(model=self.model_name,input=[text])
        #print(f"Mistral encode() response: {embeddings_response.data[0].embedding}")
        return embeddings_response.data[0].embedding
        
    def decode(self, tokens: List[int]) -> str:
        """Return the text for the tokens"""
        return self.last_text   # DIRTY hack to get the text from the last encode() call to test if Mistral API is working while I do not yet know how to decode the tokens

    def free(self) -> None:
        """Free the memory used by the embeddings"""
        self.tokenizer = None


class MistralModel:
    """This class wraps the Mistral API calls"""
    def __init__(self, setup: dict, system_info:SystemInfo, Mistral_api_key: str, model_name: str) -> None:
        self.client     = MistralClient(api_key=Mistral_api_key)
        self.model_name = model_name

    @staticmethod
    def keyify_model_id(model_id):
        """Clean the model_id into a string that can be used as a key"""
        return str(model_id).replace('/', '_-_')
    
    @staticmethod
    def de_keyify_model_id(cleaned_model_id):
        """Reverse the cleaning process to get the original model_id"""
        return cleaned_model_id.replace('_-_', '/')
    
    def predict(self, prompt: str) -> List[str]:
        """predict the next token based on the prompt"""
        messages      = [ ChatMessage(role="user", content=prompt)]
        chat_response = self.client.chat( model=self.model_name, messages=messages)
        print(f"Mistral predict() response type: {type(chat_response)}")
        for k in chat_response:                       # What a mess of an output format!
            print(f"### key: {k}, type: {type(k)}")
            if k[0] == 'choices':
                for l in k[1]:
                    print(f"#### L type: {type(l)} key: {l}, ")
                    for m in l:
                        print(f"##### M type: {type(m)} key: {m}, ")
                        if m[0] == 'message':
                            for n in m[1]:
                                print(f"###### N type: {type(n)} key: {n}, ")
                                if n[0] == 'content':
                                    print(f"####### content: {n[1]}")
                                    content = n[1]

        return content
    
    def __del__(self):
        if self is not None:
            if hasattr(self, 'client'):  del self.client
            self.free()

    def free(self) -> None:
        """Free the memory used by the model"""
        if hasattr(self, 'client'    ):  self.client     = None
        if hasattr(self, 'model_name'):  self.model_name = None