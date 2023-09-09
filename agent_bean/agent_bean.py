
#
#  conda activate AB311
#
""" 
Agent bean aims at providing a simple interface to interact with a language model. 

it allows to launch the agent action and get the response. 

and at terms it will be extended to 
 - make the link with the backlog of action 
 - make the link with the context manager
 
 """

from   dotenv                            import load_dotenv
from   agent_bean.agent_actions          import AgentAction
from   agent_bean.models_manager         import ModelsManager
from   agent_bean.system_info            import SystemInfo


class AgentBean:
  """ AgentBean is a langchain interface to collect questions and feed them to a llm """

  def __init__(self, setup: dict) -> None:
    load_dotenv()
    self.setup          = setup
    self.debug          = setup['debug']
    self.si             = SystemInfo()
    self.mm             = ModelsManager(setup, self.si            )
    self.aa             = AgentAction(  setup, self.si,  self.mm, )


  def agent_action(self, action_type: str, inputs: list) -> str:
    """prepare the prompt for a given action and call the model"""

    if self.debug:
      print(f"Action: {action_type}, num Inputs: {len(inputs)}")

    resp = self.aa.perform_action(action_type, inputs)
    print(f"ZZZ R E S P O N S E: {resp}")

    return resp


