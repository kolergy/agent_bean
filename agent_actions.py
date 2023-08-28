class AgentAction():
    """This class is used to define the actions that the agent can take."""
    def __init__(self, setup):
        self.setup          = setup
        self.actions_list   = [ m for m in self.__dir__() if m.startswith('action_') ]

    def perform_action(self, action_type, inputs):
        """Perform the action specified by action_type on the inputs."""
        action_name = f"__action_{action_type}__"
        if action_name in self.actions_list:
            return(getattr(self, action_name)(inputs))
        else:
            return("AgentAction ERROR: Action {action_name} not implemented (yet?)")


    def __action_summarize__(self, inputs):
      """Summarize the input text."""
      prompt = self.setup['prompts_templates']["summarize"].format(text=inputs[0])
      return(self.model.predict(prompt,
                                max_tokens       = 1000,
                                temperature      =    0.0,
                                top_p            =    1,
                                frequency_penalty=    0,
                                presence_penalty =    0.6,
                                stop             = ["\n"]))


    def __action_search__(self, inputs):
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
