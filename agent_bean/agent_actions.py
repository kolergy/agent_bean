
import torch

from   typing                            import List, Dict
from   langchain.tools                   import DuckDuckGoSearchResults
from   agent_bean.system_info            import SystemInfo
from   agent_bean.models_manager         import ModelsManager


class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup: dict, sys_info: SystemInfo, mm: ModelsManager) -> None:
        self.setup            = setup
        self.system_info      = sys_info
        self.search           = DuckDuckGoSearchResults()
        self.actions_list     = [ m for m in dir(self) if m.startswith('__action_') ]
        self.actions_str_list = [ m.replace('__action_', '').replace('__', '') for m in self.actions_list ]
        self.mm               = mm
        print(f"Actions list: {self.actions_list}")


    def perform_action(self, action_type: str, inputs: List[str]) -> str:
        """Perform the action specified by action_type on the inputs."""
        action_name = f"__action_{action_type}__"
        if action_name in self.actions_list:
            return(getattr(self, action_name)(inputs))
        else:
            #return([f"AgentAction ERROR: Action {action_name} not implemented (yet?)"])
            raise NotImplementedError(f"AgentAction ERROR: Action {action_name} not implemented (yet?)")


    def get_available_actions(self) -> List[str]:
        """Return the list of available actions."""
        return self.actions_str_list


    def get_special_tokens(self, model_name:str) -> [dict]:
        """get the special tokens used by the model"""
        keys = ["model_sys_delim", "model_usr_delim"]
        out  = {k:self.setup['models_list'][model_name][k] for k in keys  }
        return out

    def __action__generate__(self, action_data: Dict) -> str:
        """use a llm agent to generate text based on input prompt"""
        action_name    = 'generate'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_


    def __action_free__(self, inputs: List[str]) -> str:
        """Generate code based on the input text."""
        action_name    = 'free'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_special_tokens(model_name)

        max_tokens     = int(0.7 * self.setup['models_list'][model_name]['max_tokens'])
        prompt         = special_tokens['model_sys_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_system']) \
                            + special_tokens['model_sys_delim']['end']
        prompt        += special_tokens['model_usr_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_template']).format(text=inputs) \
                            + special_tokens['model_usr_delim']['end']
        
        self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                     'temperature':       0.6,
                                     'top_p':             1,
                                     'frequency_penalty': 0,
                                     'presence_penalty':  0.6,
                                     'stop':              ["\n"]})
        resp           = self.mm.predict(model_name, prompt)
        del action_name
        del model_name
        del special_tokens
        del max_tokens
        del prompt

        return resp
    

    def __action_summarize__(self, inputs: List[str]) -> str:
        """Summarize the input text."""
        # Tokenize the input text
        action_name    = 'summarize'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_special_tokens(model_name)
        input_tokens   = self.mm.get_embeddings(model_name, inputs[0])
        max_tokens     = int(0.7 * self.setup['models_list'][model_name]['max_tokens'])
        summaries      = []
            
        # Split the tokenized input into chunks and summarize each chunk
        for i in range(0, len(input_tokens), max_tokens):
            chunk      = input_tokens[i:i+max_tokens]
            chunk_text = str(self.mm.decode(model_name, chunk))
            #prompt     = ''.join(self.setup['actions']['summarize']['prompt_template']).format(text=chunk_text)
            prompt     = special_tokens['model_sys_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_system']) \
                            + special_tokens['model_sys_delim']['end']
            prompt    += special_tokens['model_usr_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_template']).format(text=chunk_text) \
                            + special_tokens['model_usr_delim']['end']
            #self.system_info.print_GPU_info()

            self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                     'temperature':       0.01,
                                     'top_p':             1,
                                     'frequency_penalty': 0,
                                     'presence_penalty':  0.6,
                                     'stop':              ["\n"]})
            summary = self.mm.predict(model_name, prompt)
            summaries.append(summary[-1])

        # Concatenate the summaries to form the final summary
        res = ' '.join(summaries)  # Concatenate the summaries to form the final summary
        return res


    def __action_search__(self, inputs: List[str]) -> str:
        """Search internet for the input text subject."""
        action_name    = 'search'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_special_tokens(model_name)
        max_tokens     = int(0.7 * self.setup['models_list'][model_name]['max_tokens'])
        resp           = []
        #prompt        = ''.join(self.setup['actions']['search']['prompt_template']).format(text=inputs)
        prompt         = special_tokens['model_sys_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_system']) \
                            + special_tokens['model_sys_delim']['end']
        prompt        += special_tokens['model_usr_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_template']).format(text=inputs) \
                            + special_tokens['model_usr_delim']['end']   
             
        self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                     'temperature':       0.01,
                                     'top_p':             1,
                                     'frequency_penalty': 0,
                                     'presence_penalty':  0.6,
                                     'stop':              ["\n"]})
        search_querry = self.mm.predict(model_name, prompt)
        if len(search_querry[-1]) > 0:
            search_resp   = str(self.search.run(search_querry[-1]))
        else:  # If the model did not generate a search querry, use the input text as the search querry
            search_resp   = str(self.search.run(inputs[0]))
        resp.append(search_querry[-1])
        resp.append(search_resp)
        return ' '.join(resp)


    def __action_split__(self, inputs: List[str]) -> List[str]:
        """Split a complex task into a set of simple tasks."""
        action_name    = 'split'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_special_tokens(model_name)
        max_tokens     = int(0.7 * self.setup['models_list'][model_name]['max_tokens'])
        #prompt         = ''.join(self.setup['actions']['split']['prompt_template']).format(text=inputs)
        prompt         = special_tokens['model_sys_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_system']) \
                            + special_tokens['model_sys_delim']['end']
        prompt        += special_tokens['model_usr_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_template']).format(text=inputs) \
                            + special_tokens['model_usr_delim']['end']    
               
        self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                     'temperature':       0.01,
                                     'top_p':             1,
                                     'frequency_penalty': 0,
                                     'presence_penalty':  0.6,
                                     'stop':              ["\n"]})
        resp         = self.mm.predict(model_name, prompt)
        tasks        = resp[0].split(',')
        return tasks


    def __action_code__(self, inputs: List[str]) -> str:
        """Generate code based on the input text."""
        action_name    = 'code'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_special_tokens(model_name)
        max_tokens     = int(0.7 * self.setup['models_list'][model_name]['max_tokens'])
        #prompt         = ''.join(self.setup['actions']['code']['prompt_template']).format(text=inputs)
        prompt         = special_tokens['model_sys_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_system']) \
                            + special_tokens['model_sys_delim']['end']
        prompt        += special_tokens['model_usr_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_template']).format(text=inputs) \
                            + special_tokens['model_usr_delim']['end']   
        
        self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                     'temperature':       0.4,
                                     'top_p':             1,
                                     'frequency_penalty': 0,
                                     'presence_penalty':  0.6,
                                     'stop':              ["\n"]})
        resp         = self.mm.predict(model_name, prompt)
        code         = resp
        return code


    def __action_code_quality__(self, inputs: List[str]) -> str:
        """Check the quality of the input code."""
        action_name    = 'code_quality'
        model_name     = self.setup['actions'][action_name]['model_name']
        special_tokens = self.get_special_tokens(model_name)
        max_tokens     = int(0.7 * self.setup['models_list'][model_name]['max_tokens'])
        #prompt        = ''.join(self.setup['actions']['code_quality']['prompt_template']).format(text=inputs)
        prompt         = special_tokens['model_sys_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_system']) \
                            + special_tokens['model_sys_delim']['end']
        prompt        += special_tokens['model_usr_delim']['start'] \
                            + ''.join(self.setup['actions'][action_name]['prompt_template']).format(text=inputs) \
                            + special_tokens['model_usr_delim']['end']   
        
        self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                     'temperature':       0.5,
                                     'top_p':             1,
                                     'frequency_penalty': 0,
                                     'presence_penalty':  0.6,
                                     'stop':              ["\n"]})
        resp         = self.mm.predict(model_name, prompt)
        code_quality = resp
        return code_quality


    def __del__(self):
        """Delete the agent action."""
        print("---- Deleting AgentAction ----")
        if self is not None:
            if hasattr(self, 'mm'):               del self.mm
            if hasattr(self, 'setup'):            del self.setup
            if hasattr(self, 'system_info'):      del self.system_info
            if hasattr(self, 'actions_str_list'): del self.actions_str_list
            if hasattr(self, 'actions_list'):     del self.actions_list
            if hasattr(self, 'search'):           del self.search
        if torch.cuda is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
