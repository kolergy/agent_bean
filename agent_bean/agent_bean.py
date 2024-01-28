
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
from   typing                            import Dict
from   dotenv                            import load_dotenv
from   tinydb                            import TinyDB, Query

from   agent_bean.agent_actions          import AgentAction
from   agent_bean.models_manager         import ModelsManager
from   agent_bean.system_info            import SystemInfo
from   agent_bean.chat_content           import ChatContent

class AgentBean:
  """ AgentBean is an interface to collect questions and feed them to a llm """

  def __init__(self, setup: dict) -> None:
    load_dotenv()       # to get API keys if any
    self.setup          = setup
    self.debug          = setup['debug'       ]
    self.db_file_name   = setup['db_file_name']
    self.si             = SystemInfo()
    self.mm             = ModelsManager(setup, self.si            )
    self.aa             = AgentAction(  setup, self.si,  self.mm, )
    self.db             = TinyDB(       'agent_db.json'           )
    self.chat_content   = ChatContent(  self.mm                   )

  def setup_update(self, setup: dict) -> None:
    """update the setup and propagate it to the other components"""
    self.setup = setup
    self.debug = setup['debug']
    self.mm.setup_update(setup)
    self.aa.setup_update(setup)


  def agent_action(self, action_name: str, inputs: list) -> str:
    """Execute a given action and store the inputs and outputs in the db"""
    if self.debug:
      print(f"Action: {action_name}, num Inputs: {len(inputs)}")
    context    = self.chat_content.get_context()
    if self.debug:
      print(f"ZZZ Context: {context}")
    ctx_input  = context + inputs
    resp       = self.aa.perform_action(action_name, ctx_input)
    model_name = self.setup["actions"][action_name]["model_name"]
    print(f"ZZZ R E S P O N S E: {resp}")
    self.chat_content.add_interaction(
          action_name = action_name, 
          model_name  = model_name, 
          input_text  = inputs, 
          context     = context,
          output_text = resp)
    task = {"action":action_name, "inputs":inputs, "responce":resp}
    self.db.insert(task)
    return resp

  def action_list(self, list_of_a:[Dict]) -> None:
    """execute a list of actions"""
    for a in list_of_a: 
      print(f"Action: {a}")
      self.agent_action(action_name=a["action_name"], inputs=a["inputs"])
