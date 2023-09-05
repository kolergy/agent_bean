
import torch

from   typing                            import List
from   langchain.tools                   import DuckDuckGoSearchResults
from   system_info                       import SystemInfo
from   models_manager                    import ModelsManager


class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup: dict, sys_info: SystemInfo, mm: ModelsManager) -> None:
        self.setup          = setup
        self.system_info    = sys_info
        self.search         = DuckDuckGoSearchResults()
        self.actions_list   = [ m for m in dir(self) if m.startswith('__action_') ]
        self.mm             = mm
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
        model_name   = self.setup['model']['name']
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
            summary    = self.mm.predict(model_name, prompt,
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
        model_name   = self.setup['model']['name']
        resp = []
        prompt        = ''.join(self.setup['prompts_templates']["search"]).format(text=inputs[0])
        search_querry = self.mm.predict(model_name, prompt,
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