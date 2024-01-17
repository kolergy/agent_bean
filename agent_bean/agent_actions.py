
import torch

from   typing                            import List, Dict
from   langchain_community.tools         import DuckDuckGoSearchResults
from   agent_bean.system_info            import SystemInfo
from   agent_bean.models_manager         import ModelsManager
from   agent_bean.file_loader            import FileLoader

class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup: dict, sys_info: SystemInfo, mm: ModelsManager) -> None:

        self.setup_update(setup)
        self.system_info        = sys_info
        self.search             = DuckDuckGoSearchResults()
        self.actions_list       = [ m for m in dir(self) if m.startswith('__action_') ]
        self.actions_str_list   = [ m.replace('__action_', '').replace('__', '') for m in self.actions_list ]
        self.functions_list     = [ m for m in dir(self) if m.startswith('__function_') ]
        self.functions_str_list = [ m.replace('__function_', '').replace('__', '') for m in self.functions_list ]
        self.mm                 = mm
        print(f"      Actions list: {self.get_available_actions()}")
        print(f"Actions types list: {self.actions_list}")
        print(f"    Functions list: {self.functions_list}")


    def setup_update(self, setup: dict) -> None:
        """update the setup """
        self.setup = setup



    def perform_action(self, action_name: str, inputs: List[str]) -> str:
        """Perform the action type specified using the params in the settings file and the input. this function returns the response."""
        action_params:Dict      = self.setup["actions"][action_name].copy()
        action_params["inputs"] = inputs
        action_type:str         = action_params["action_type"]
        action_f_name:str       = f"__action_{action_type}__"
        
        if action_f_name in self.actions_list:
            return(getattr(self, action_f_name)(action_params))
        else:
            raise NotImplementedError(f"AgentAction ERROR: Action {action_f_name} not implemented (yet?)")


    def perform_function(self, function_name: str, input:str) -> str:
        """Perform the function specified using the input. this function returns the output string."""
        function_f_name:str = f"__function_{function_name}__"
        if function_f_name in self.functions_list:
            return(getattr(self, function_f_name)(input))
        else:
            raise NotImplementedError(f"AgentAction ERROR: Action Function {function_f_name} not implemented (yet?)")


    def get_available_actions(self) -> List[str]:
        """Return the list of available actions."""
        return self.setup["actions"].keys()
        

    def get_available_actions_types(self) -> List[str]:
        """Return the list of available actions."""
        return self.actions_str_list
        

    def get_available_functions(self) -> List[str]:
        """Return the list of available functions."""
        return self.functions_str_list
    

    def get_special_tokens(self, model_name:str) -> [dict]:
        """get the special tokens used by the model"""
        keys = ["model_sys_delim", "model_usr_delim"]
        out  = {k:self.setup['models_list'][model_name][k] for k in keys  }
        return out
        

    def __function_search__(self, input:str) -> str:
        """search for the input text using duckduckgo"""
        return self.search.run(input)

    def __function_load_file__(self, input:str) -> str:
        """load the file specified in the input"""
        return FileLoader.load_file(input)

    def __action_generate__(self, action_params) -> str:
        """use a llm agent to generate text based on input prompt"""
        ##action_name    = kwargs['action_name']
        model_name           = action_params['model_name' ]
        inputs               = action_params['inputs'     ].copy()
        action_model_params  = action_params.get('model_params', {})
        action_pre_function  = action_params.get('action_pre_function' ,  None   )
        action_post_function = action_params.get('action_post_function',  None   )
        code_language        = action_params.get('code_language'       , 'python')

        if action_pre_function is not None and len(inputs) > 0:
            function_res:str    = self.perform_function(action_pre_function, inputs.pop(0))
            inputs.insert(0, function_res)
            
        special_tokens   = self.get_special_tokens(model_name)
        max_tokens       = int(action_params.get('max_token_ratio', 0.7) * self.setup['models_list'][model_name]['max_tokens'])
        input_tokens     = self.mm.get_embeddings(model_name, inputs[0])
        resps            = []
        chunkable_action = action_params.get('chunkable_action', False)
        if len(input_tokens) > max_tokens and not chunkable_action:
            raise ValueError(f"AgentAction ERROR: input text too long for model {model_name}")
        
        while True:
            token_index    = min(len(input_tokens), max_tokens)
            chunk          = input_tokens[0:token_index].copy()       # Copy the chunk of tokens to avoid modifying the input tokens
            input_tokens   = input_tokens[token_index: ]              # Remove the chunk from the input tokens
            chunk_text     = str(self.mm.decode(model_name, chunk))

            prompt         = special_tokens['model_sys_delim']['start'] 
            prompt        += ''.join(action_params['prompt_system']).format(code_language=code_language) 
            prompt        += special_tokens['model_sys_delim']['end']
            prompt        += special_tokens['model_usr_delim']['start'] 
            prompt        += ''.join(action_params['prompt_template']).format(text=chunk_text, code_language=code_language) 
            prompt        += special_tokens['model_usr_delim']['end']

            """
            self.mm.set_model_params(model_name, params={'max_tokens':       max_tokens,
                                        'temperature':       action_params.get('temperature'      ,   1.0  ),
                                        'top_p':             action_params.get('top_p'            ,   1.0  ),
                                        'frequency_penalty': action_params.get('frequency_penalty',   0.5  ),
                                        'presence_penalty':  action_params.get('presence_penalty' ,   0.1  ),
                                        'max_new_tokens':    action_params.get('max_new_tokens'   , 512    ),
                                        'stop':              action_params.get('stop'             , ["\n"] )})
            """
            self.mm.set_model_params(model_name, params=action_model_params)
            
            resp           = self.mm.predict(model_name, prompt).copy()

            if chunkable_action:              # If the action is chunkable, 
                resps.append(resp[-1])        # we need to keep adding chunks to the response  
                if len(input_tokens) == 0:
                    break                     # until we reach the end of the input tokens

            else:                             # If the action is not chunkable, no need to loop
                break

        if chunkable_action:
            resp = ' '.join(resps)  # Concatenate the chunks to form the final response

        if action_post_function is not None:
            resp = self.perform_function(action_post_function, resp[-1])

        output_type = action_params.get('output_type', 'text')
        if output_type == 'text':
            return resp

        if output_type == 'code_text':
            # TODO 1. remove the text before the code
            
            # TODO 2. remove the code delimiters tokens
            pass

        if output_type == 'actions_json':
            # TODO 1. remove the text before the json

            # TODO 2. remove the json delimiters tokens 

            # TODO 3. convert the json to a dictionary as a list of actions and append to the backlog
            pass

        return resp


