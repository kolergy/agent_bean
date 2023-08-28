from typing import List, Any

class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup: dict, encoder: Any) -> None:
        self.setup          = setup
        self.enc            = encoder
        self.actions_list   = [ m for m in self.__dir__() if m.startswith('action_') ]

    def perform_action(self, action_type: str, inputs: List[str]) -> str:
        """Perform the action specified by action_type on the inputs."""
        action_name = f"__action_{action_type}__"
        if action_name in self.actions_list:
            return(getattr(self, action_name)(inputs))
        else:
            return("AgentAction ERROR: Action {action_name} not implemented (yet?)")


    def __action_summarize__(self, inputs: List[str]) -> str:
        """Summarize the input text."""
        # Tokenize the input text
        input_tokens = self.enc.encode(inputs[0])
        max_tokens   = int(0.8 * self.setup['model']['max_tokens'])
        summaries    = []

        # Split the tokenized input into chunks and summarize each chunk
        for i in range(0, len(input_tokens), max_tokens):
            chunk      = input_tokens[i:i+max_tokens]
            chunk_text = self.enc.decode(chunk)
            prompt     = self.setup['prompts_templates']["summarize"].format(text=chunk_text)
            summary    = self.model.predict(prompt,
                                            max_tokens       = max_tokens,
                                            temperature      = 0.0,
                                            top_p            = 1,
                                            frequency_penalty= 0,
                                            presence_penalty = 0.6,
                                            stop             = ["\n"])
            summaries.append(summary)

        # Concatenate the summaries to form the final summary
        return ' '.join(summaries)


    def __action_search__(self, inputs: List[str]) -> List[str]:
        """Search internet for the input text subject."""
        prompt    = self.setup['prompts_templates']["search"].format(text=inputs[0])
        responses = self.model.predict(prompt,
                                       max_tokens       = 1000,
                                       temperature      =    0.0,
                                       top_p            =    1,
                                       frequency_penalty=    0,
                                       presence_penalty =    0.6,
                                       stop             = ["\n"])
        responses.append(self.search.run(responses[-1]))
        return(responses)
