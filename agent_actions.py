import torch

from typing          import List, Any
from langchain.tools import DuckDuckGoSearchResults
from system_info     import SystemInfo

class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup: dict, model: Any, encoder: Any, sys_info: SystemInfo) -> None:
        self.setup          = setup
        self.enc            = encoder
        self.model          = model
        self.system_info    = sys_info
        self.search         = DuckDuckGoSearchResults()
        self.actions_list   = [ m for m in dir(self) if m.startswith('__action_') ]
        print(f"Actions list: {self.actions_list}")

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
            prompt     = self.setup['prompts_templates']["summarize"].format(text=chunk_text)
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
        prompt        = self.setup['prompts_templates']["search"].format(text=inputs[0])
        search_querry = self.model.predict(prompt,
                                       max_tokens       = 1000,
                                       temperature      =    0.01,
                                       top_p            =    1,
                                       frequency_penalty=    0,
                                       presence_penalty =    0.6,
                                       stop             = ["\n"])
        #print(f"Search search_querry type: {type(search_querry[-1])}, len: {len(search_querry)}")
        #print(f"Search search_querry: {search_querry}")
        resp.append(search_querry[-1])
        search_resp = str(self.search.run(search_querry[-1]))
        #print(f"Search response type: {type(search_resp)}, len: {len(search_resp)}")
        #print(f"Search response: {search_resp}")
        resp.append(search_resp)
        #print(f"Search resp type: {type(resp)}, len: {len(resp)}")
        #print(f"Search resp: {resp}")
        return ' '.join(resp)

    
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