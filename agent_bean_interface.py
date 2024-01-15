import sys
import time

import pandas                 as     pd
import gradio                 as     gr

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


class Logger:
    """ Class to redirect the console output to a file to enable its display in the interface"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False 
    
sys.stdout = Logger("output.log")
sys.stderr = Logger("output.log")

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()


# Load the settings json file and create a AgentBean object
res = FileLoader.load_json_file(settings_file)
if res['json_content'] is not None:
    setup = res['json_content']
    agent = AgentBean(setup)
else:
    print(f"ERROR: Could not load the settings file: {settings_file}")
    
default_action = "team_manager_speach"
default_model  = setup["actions"][default_action]["model_name"]

cpu_brand      =  agent.si.get_cpu_brand(   )
cpu_cores      =  agent.si.get_cpu_cores(   )
ram_total_Gb   =  agent.si.get_ram_total(   )
gpu_brand      =  agent.si.get_gpu_brand(   )
v_ram_total_Gb =  agent.si.get_v_ram_total( )

ram_used_Gb    = [agent.si.get_ram_used()   ]
v_ram_used_Gb  = [agent.si.get_v_ram_used() ]
time_s         = [0.0                       ]
start_time     =  time.time()

ram_label      = f"CPU: {cpu_brand}, {cpu_cores} Cores, RAM Total: {ram_total_Gb:6.2f} Gb, RAM Used: {agent.si.get_ram_used():6.2f} Gb, RAM Free: {agent.si.get_ram_free():6.2f} Gb"
v_ram_label    = f"GPU: {gpu_brand}, VRAM Total: {v_ram_total_Gb:6.2f} Gb, VRAM Used: {agent.si.get_v_ram_used():6.2f} Gb, VRAM Free: {agent.si.get_v_ram_free():6.2f} Gb"


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
        

# Define the function to be called when the Run button is pressed
def run_action(action_name, action_input):
    """ Run the action: using the configured llm agent"""
    output = ''.join(agent.agent_action(action_name, [action_input]))
    print(f"Action Completed LEN OUTPUT: {len(output)}")
    return output


def update_ram():
    """ Update the ram plot """
    #print(f"update_ram() called, elapsed: {time.time() - start_time:6.2f} s. ram_used_Gb: {agent.si.get_ram_used():6.2f} Gb.")
    ram_used_Gb.append(  agent.si.get_ram_used()   )
    v_ram_used_Gb.append(agent.si.get_v_ram_used() )
    time_s.append(       time.time() - start_time  )

    df         = pd.DataFrame({'time_s': time_s, 'ram_used_Gb': ram_used_Gb, 'v_ram_used_Gb': v_ram_used_Gb})
    update_ram = gr.LinePlot(
                        value  = df, 
                        title  = ram_label, 
                        x      = 'time_s', 
                        y      = 'ram_used_Gb', 
                        y_lim  = [0.0, ram_total_Gb], 
                        height =  250, 
                        width  = 1000)

    return update_ram

def update_v_ram():
    """ Update the v_ram plot """
    #print(f"update_v_ram() called, elapsed: {time.time() - start_time:6.2f} s. v_ram_used_Gb: {agent.si.get_v_ram_used():6.2f} Gb. v_ram_total_Gb: {agent.si.get_v_ram_total():6.2f} Gb. v_ram_free_Gb: {agent.si.get_v_ram_free():6.2f} Gb.")
    df         = pd.DataFrame({'time_s': time_s, 'ram_used_Gb': ram_used_Gb, 'v_ram_used_Gb': v_ram_used_Gb})  
    update_v_ram = gr.LinePlot(
                        value  = df, 
                        title  = v_ram_label, 
                        x      = 'time_s', 
                        y      = 'v_ram_used_Gb',  
                        y_lim  = [0.0, v_ram_total_Gb],
                        height =  250, 
                        width  = 1000)

    return update_v_ram

def update_action_name(action_name):
    """Get the default model for the selected action"""
    default_model = setup["actions"][action_name]["model_name"]
    # Return a new Dropdown object with the default model selected
    print(f"ZZZZZZZZZ default model:{default_model}\n{agent.mm.get_available_models()}\nZZZZZZZZZ")
    return gr.components.Dropdown(choices=agent.mm.get_available_models() , label="Model Name" , value=default_model  )

def update_model_name(action_name, model_name):
    """update the settings with the selected model for the selected action"""
    setup["actions"][action_name]["model_name"] = model_name
    print(f"Model changed during this sessio for action: {action_name} to this model:{model_name}")
 
# Define the Gradio display
with gr.Blocks(title="Agent Bean Interface") as iface:
    gr.Markdown("# Agent Bean Interface  ")
    with gr.Row():
        action_name      = gr.components.Dropdown(choices=agent.aa.get_available_actions(), label="Action Name", value=default_action, interactive=True)
        model_name       = gr.components.Dropdown(choices=agent.mm.get_available_models() , label="Model Name" , value=default_model, interactive=True )
        # Link the action_name dropdown to the update_model_name function
        action_name.change(update_action_name, inputs=[action_name], outputs=[model_name])
        model_name.change( update_model_name , inputs=[action_name,model_name])

    with gr.Row():
        action_input     = gr.components.Textbox( lines   = 2,  autoscroll = True , label = "Action Input", scale=1   )
        run_button       = gr.Button(             variant = 'primary'             , value = "Run Agent"   , scale=0   )
    # Removed the duplicate render() calls
    text_output  = gr.components.Textbox( lines   = 6,  autoscroll = True, label = "Output Text"    )
    
    with gr.Row():
        actions_list     = gr.components.File(file_count=1, file_types=["json"], value=a_list_file, label = "Actions List File"     )
        run_list_button  = gr.Button(             variant = 'secondary'           , value = "Run Actions List"  )

    with gr.Row():
        console_output = gr.components.Textbox(interactive=False, label="Console Output", lines=4, autoscroll=True, value="Console will display here...")

    with gr.Row():
        ram_plt          = gr.components.LinePlot(show_label=False)
        v_ram_plt        = gr.components.LinePlot(show_label=False)

    run_list_button.click( run_list_action, actions_list)
    run_button.click( run_action, [action_name, action_input], outputs = text_output)

    dep_ram      = iface.load(update_ram  , None, ram_plt  , every=1)
    dep_v_ram    = iface.load(update_v_ram, None, v_ram_plt, every=1)

    iface.load(read_logs, None, console_output, every=1)

# Launch the interface
iface.queue().launch(share=False)
