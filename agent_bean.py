import os
import json
import tiktoken

from   dotenv                            import load_dotenv
from   langchain.chat_models             import ChatOpenAI
from   langchain.vectorstores            import FAISS
from   langchain.tools                   import DuckDuckGoSearchResults
from   langchain.document_loaders        import TextLoader
from   langchain.embeddings.openai       import OpenAIEmbeddings
from   langchain.text_splitter           import CharacterTextSplitter
from   agent_actions                     import AgentAction

class AgentBean:
  """ LangIf is a langchain interface to collect questions and feed them to a llm """

  def __init__(self, setup: dict) -> None:
    load_dotenv()
    self.setup          = setup
    self._context       = []
    self._context_tok   = []  # new attribute to store tokenized context
    self._context_n_tok = []
    self.search         = DuckDuckGoSearchResults()
    self.debug          = setup['debug']
    self.instantiate_model()
    self.instantiate_vectorstore()


  def add_context(self, context_elements: list) -> None:
    """Add an array of context elements to the current context"""
    self._context.extend(context_elements)
    new_ctx_tok = [self.enc.encode(c) for c in context_elements]
    self._context_tok.extend(new_ctx_tok)  # add tokenized context
    self._context_n_tok.extend([len(tok) for tok in new_ctx_tok])  # add number of tokens


  def clear_context(self) -> None:
    """Clear the context"""
    self._context       = []
    self._context_tok   = []
    self._context_n_tok = []


  def instantiate_model(self) -> None:
    """instantiate the model defined in the set-up """
    if self.setup['model']['model_type'] == "openAI":
      api_key         = os.getenv('OPENAI_API_KEY')
      self.model      = ChatOpenAI(openai_api_key=api_key, model_name=self.setup['model']['model_id'])
      self.enc        = tiktoken.encoding_for_model(self.setup['model']['model_id'])
      self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    elif self.setup['model']['model_type'] == "transformers":
      # Instantiate the Transformers model here
      # You will need to fill in the details based on how you want to use the Transformers library
      pass

    self.actions        = AgentAction(self.setup, self.enc)
    
    if self.debug:
      print(f"Model initiated, type: {self.setup['model']['model_type']}, id: {self.setup['model']['model_id']}")


  def instantiate_vectorstore(self) -> None:
    """instantiate the vectorstore defined in the set-up """
    if self.setup['vectorstore']['type'] == "faiss":
      vs_path = self.setup['vectorstore']['path']
      if not os.path.isdir(vs_path): # create vect store dir if it dose not exists
        os.mkdir(vs_path)
      if not os.path.isfile(vs_path + "/index.faiss"):  # create vect store file if it dose not exists
        self.v_db = FAISS.from_texts([""], self.embeddings)
        self.v_db.save_local(vs_path)
      else:
        self.v_db = FAISS.load_local(vs_path, self.embeddings)

    if self.debug:
      print(f"Vectorstore initiated, type: {self.setup['vectorstore']['type']}")
    self.text_splitter = CharacterTextSplitter(
                            chunk_size    = self.setup['vectorstore']['chunk_size'],
                            chunk_overlap = self.setup['vectorstore']['chunk_overlap'])


  def manage_context_length(self) -> None:
    """Ensure the total number of tokens in the context is below max_tokens. If not, summarize it."""
    max_tokens = self.setup['model']['max_tokens']
    while sum(self._context_n_tok) > max_tokens:
      # Summarize the first context element and replace it in the context
      summarized_context         = self.actions.perform_action('summarize', [self._context[0]])
      summarized_context_encoded = self.enc.encode(summarized_context)

      self._context        = [summarized_context]
      self._context_tok    = [summarized_context_encoded]
      self._context_n_tok  = [len(summarized_context_encoded)]

  def agent_action(self, action_type: str, inputs: list) -> str:
    """prepare the prompt for a given action and call the model"""
    inputs_encoded    = []
    inputs_n_tok      = []
    responses         = []
    responses_encoded = []
    responses_n_tok   = []
    i = 0
    self.v_db.from_texts(inputs, self.embeddings)
    for ip in inputs:
      inputs_encoded.append(self.enc.encode(ip))
      inputs_n_tok.append(len(inputs_encoded[-1]))
      print(f"Input[{i}] number of tokens: {inputs_n_tok[-1]}")
      i += 1
    self.manage_context_length()
    if self.debug:
      print(f"Action: {action_type}, sum of Inputs: {sum(inputs_n_tok)}, Context n tok: {sum(self._context_n_tok)}")

    responses.append(self.actions.perform_action(action_type, inputs))

    print(f"Responses: {responses}")
    self.v_db.from_texts(responses, self.embeddings)
    for res in responses:
      responses_encoded.append(self.enc.encode(res))
      responses_n_tok.append(len(responses_encoded[-1]))
    self._context.extend(inputs)
    self._context_n_tok.extend(inputs_n_tok)
    self._context.extend(responses)
    self.v_db.save_local(self.setup['vectorstore']['path'])
    return responses[-1]


  def load_document(self, files_path: list) -> None:
    """Loads a set of documents in the vectorstore"""
    for file_path in files_path:
      if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise Exception(f"File {file_path} does not exist or is empty.")
      raw_documents = TextLoader(file_path).load()
      documents     = self.text_splitter.split_documents(raw_documents)
      self.v_db.from_documents(documents, OpenAIEmbeddings())
      self.v_db.save_local(self.setup['vectorstore']['path'])
