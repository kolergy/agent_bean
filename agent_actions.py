import os
import json
import torch
import tiktoken

from   typing                            import List
from   transformers_model                import TfModel
from   langchain.chat_models             import ChatOpenAI
from   langchain.embeddings.openai       import OpenAIEmbeddings
from   langchain.tools                   import DuckDuckGoSearchResults
from   system_info                       import SystemInfo

class ModelManager():
    def __init__(self, setup, debug) -> None:
        self.setup             = setup
        self.system_info       = SystemInfo()
        self.debug             = debug
        self.active_models     = {}
        self.active_embeddings = {}
        if os.path.exists(self.setup["known_models_file_name"]):
            with open(self.setup["known_models_file_name"]) as f:
                self.known_models  = json.load(f)
        else:
            self.known_models      = {}
    
    
    def model_need(self, model_name:str):
        if model_name not in self.active_models:
            self.require_instantiaton(model_name)
        

    def test_models_ressources_reqs(self):
        if self.debug:
            print(f"Testing models memory ressources requirements")
            self.system_info.print_GPU_info()
        for model_name in self.setup["models_list"].keys():
            if model_name["model_id"] not in self.known_models.keys():
                self.known_models[model_name]["model_id"] = {}
                if self.debug:
                    print(f"Model {model_name}: is not yet known Testing model memory ressources requirements")
                    self.system_info.print_GPU_info()
                    
                if self.setup["models_list"][model_name]["model_type"] == "transformers":
                    ram_b4   = self.system_info.get_ram_free()
                    v_ram_b4 = self.system_info.get_v_ram_free()
                    self.instantiate_model(model_name)
                    
                    delta_ram_gb   = max(0, ram_b4   - self.system_info.get_ram_free()  ) # min value is 0
                    delta_v_ram_gb = max(0, v_ram_b4 - self.system_info.get_v_ram_free()) # to avoid noise on the unused ram
                    self.known_models[model_name]["system_ram_gb"] = delta_ram_gb
                    self.known_models[model_name]["GPU_ram_gb"   ] = delta_v_ram_gb
                    if self.debug:
                        print(f"Model {model_name} instantiated using: {delta_ram_gb} Gb of system RAM and: {delta_v_ram_gb} Gb of V RAM on the GPU")
                        self.system_info.print_GPU_info()

                    self.deinstantiate_model(model_name)

                elif self.setup["models_list"][model_name]["model_type"] == "openAI":
                    self.known_models[model_name]["v_ram_need"] = 0.0
                    self.known_models[model_name]["ram_need"  ] = 0.0

                else:
                    print(f"ERROR: Unknown model type: {self.setup['models_list'][model]['model_type']}")

        #need to write the known modelt to the file at the end


    def instantiate_model(self, model_name) -> None:
        """instantiate the model defined in the set-up """
        if self.setup['model']['model_type'] == "openAI":
            api_key         = os.getenv('OPENAI_API_KEY')
            self.active_models[model_name] ChatOpenAI(openai_api_key=api_key, model_name=self.setup['model']['model_id'])
            self.enc        = tiktoken.encoding_for_model(self.setup['model']['model_id'])
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        elif self.setup['model']['model_type'] == "transformers":
            if hasattr(self, "model"):      del self.model        # fighting with a memory leak
            if hasattr(self, "enc"):        del self.enc
            if hasattr(self, "embeddings"): del self.embeddings
            if torch.cuda.is_available():
                print("-°---Emptying CUDA cache----")
                torch.cuda.empty_cache()

            print(f"GPU state before model instantiation: {torch.cuda.is_available()}")
            self.system_info.print_GPU_info()
            self.model      = TfModel(self.setup, self.system_info)
            self.enc        = self.model.tokenizer
            self.embeddings = self.model.embeddings
            print(f"GPU state after model instantiation: {torch.cuda.is_available()}")
            self.system_info.print_GPU_info()
    
    def deinstantiate_model(self, model_name):
        del self.embeddings

        if torch.cuda.is_available():
            print("-m--Emptying CUDA cache----")
            torch.cuda.empty_cache()
        pass

class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup: dict, sys_info: SystemInfo) -> None:
        self.setup          = setup
        self.system_info    = sys_info
        self.search         = DuckDuckGoSearchResults()
        self.actions_list   = [ m for m in dir(self) if m.startswith('__action_') ]
        print(f"Actions list: {self.actions_list}")
        self.instantiate_model()

    def perform_action(self, action_type: str, inputs: List[str]) -> str:
        """Perform the action specified by action_type on the inputs."""
        action_name = f"__action_{action_type}__"
        if action_name in self.actions_list:
            return(getattr(self, action_name)(inputs))
        else:
            return([f"AgentAction ERROR: Action {action_name} not implemented (yet?)"])


    def __action_summarize__(self, inputs: List[str]) -> str:
        """Summarize the input text."""
        # Tokenize the input text
        input_tokens = self.enc.encode(inputs[0])
        max_tokens   = int(0.7 * self.setup['model']['max_tokens'])
        summaries    = []
        tot_input    = ' '.join(inputs)
        print(f"AAA input_tokens len: {len(input_tokens)} max_tokens: {max_tokens}")
        #print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA tot_input:\n{tot_input}\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            
        # Split the tokenized input into chunks and summarize each chunk
        for i in range(0, len(input_tokens), max_tokens):
            chunk      = input_tokens[i:i+max_tokens]
            chunk_text = self.enc.decode(chunk)
            prompt     = ''.join(self.setup['prompts_templates']["summarize"]).format(text=chunk_text)
            self.system_info.print_GPU_info()
            print(f"BBB--- CHUNK LEN: {len(chunk)}, chunk_text len: {len(chunk_text)}, prompt len: {len(prompt)}")
            #print(f"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB chunk_text:\n{chunk_text}\nBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
            summary    = self.model.predict(prompt,
                                            max_tokens       = max_tokens,
                                            temperature      = 0.01,
                                            top_p            = 1,
                                            frequency_penalty= 0,
                                            presence_penalty = 0.6,
                                            stop             = ["\n"])
            summaries.append(summary[-1])
            #print(f"summary response type: {type(summary[-1])}, len: {len(summary)}")
        #print(f"summarized response type: {type(summaries)}")
        # Concatenate the summaries to form the final summary
        res = ' '.join(summaries)
        #print(f"summarized response type: {type(res)}, len: {len(res)}")
        #print(f"summarized response: {res}")
        return res


    def __action_search__(self, inputs: List[str]) -> str:
        """Search internet for the input text subject."""
        resp = []
        prompt        = ''.join(self.setup['prompts_templates']["search"]).format(text=inputs[0])
        search_querry = self.model.predict(prompt,
                                       max_tokens       = 1000,
                                       temperature      =    0.01,
                                       top_p            =    1,
                                       frequency_penalty=    0,
                                       presence_penalty =    0.6,
                                       stop             = ["\n"])
        resp.append(search_querry[-1])
        search_resp = str(self.search.run(search_querry[-1]))
        resp.append(search_resp)
        return ' '.join(resp)

    def __action_split__(self, inputs: List[str]) -> List[str]:
        """Split a complex task into a set of simple tasks."""
        # Here you can implement the logic to split the complex task
        # For the sake of this example, let's assume the input is a string with tasks separated by commas
        tasks = inputs[0].split(',')
        return tasks

    def __action_code__(self, inputs: List[str]) -> str:
        """Generate code based on the input text."""
        # Here you can implement the logic to generate code based on the input text
        # For the sake of this example, let's assume the input is a string describing the code to be generated
        code = inputs[0]  # replace this with your code generation logic
        return code

    def __action_code_quality__(self, inputs: List[str]) -> str:
        """Check the quality of the input code."""
        # Here you can implement the logic to check the quality of the input code
        # For the sake of this example, let's assume the input is a string containing the code to be checked
        code_quality = inputs[0]  # replace this with your code quality checking logic
        return code_quality

    def __del__(self):
        """Delete the model."""
        del self.model
        del self.enc
        del self.setup
        del self.actions_list
        del self.search
        if torch.cuda.is_available():
            print("-a--Emptying CUDA cache----")
            torch.cuda.empty_cache()
