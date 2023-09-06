
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

import gc
import torch

from   dotenv                            import load_dotenv
from   agent_bean.agent_actions          import AgentAction

class AgentBean:
  """ LangIf is a langchain interface to collect questions and feed them to a llm """

  def __init__(self, setup: dict) -> None:
    load_dotenv()
    self.setup          = setup
    self.debug          = setup['debug']
    self.aa             = AgentAction(setup, self.mm)


  def agent_action(self, action_type: str, inputs: list) -> str:
    """prepare the prompt for a given action and call the model"""

    if self.debug:
      print(f"Action: {action_type}, num Inputs: {len(inputs)}")

    resp = self.aa.perform_action(action_type, inputs)

    return resp[-1]


  def __del__(self):
    """Destroy the agent and free all memory occupied by the agent, including VRAM."""
    print(f"B4 Agent elimination CLEAN Vram: {self.system_info.get_vram_total():5.2f} GB, available: {self.system_info.get_vram_available():5.2f} GB")
      
    del self.aa
    del self.debug
    del self.setup
    gc.collect()
    # Empty the CUDA cache
    if torch.cuda.is_available():
        print("+----Emptying CUDA cache----")
        torch.cuda.empty_cache()
    print(f"AFTER CLEAN Vram: {self.system_info.get_vram_total():5.2f} GB, available: {self.system_info.get_vram_available():5.2f} GB")
