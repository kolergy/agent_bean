import sys
import time

from   agent_bean.agent_bean  import AgentBean
from   agent_bean.file_loader import FileLoader

settings_file  = 'settings.json'
a_list_file    = 'actions_list.json'

ram_total_Gb   = 0.0
v_ram_total_Gb = 0.0

ram_used_Gb    = [0.0]
v_ram_used_Gb  = [0.0]
time_s         = [0.0]
start_time     = time.time()

# Define the function to be called when the setup button is pressed
def run_list_action(actions_list):
    """ load and run the action list: using the configured llm agent"""
    type_f = type(actions_list)
    if type_f == list:             # How can someone decide to return a list or an object?
        a_list_file = actions_list[0]
    #print(f"file: {actions_list.name}")
    a_list_file = FileLoader.load_json_file(a_list_file)
    if a_list_file['json_content'] is not None:
        a_list = a_list_file['json_content'] 
        print(f"Loaded actions list: {a_list}")
        agent.action_list(a_list)
    else:
        print(f"ERROR: Could not load the actions file: {actions_list.name}, no json content")
        
# Load the settings json file and create a AgentBean object
res = FileLoader.load_json_file(settings_file)
if res['json_content'] is not None:
    setup = res['json_content']
    agent = AgentBean(setup)
else:
    print(f"ERROR: Could not load the settings file: {settings_file}")


run_list_action(a_list_file)