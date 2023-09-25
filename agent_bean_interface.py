import json
import time
import datetime
import threading

import numpy                 as     np 
import matplotlib.pyplot     as     plt
import pandas                as     pd
import gradio                as     gr

from   agent_bean.agent_bean import AgentBean


#settings_file  = 'settings_opai.json'
settings_file  = 'settings_trans.json'

ram_total_Gb   = 0.0
v_ram_total_Gb = 0.0

ram_used_Gb    = [0.0]
v_ram_used_Gb  = [0.0]
time_s         = [0.0]
start_time     = time.time()

# Load the settings json file and create a AgentBean object
with open(settings_file) as f: setup = json.load(f)
agent = AgentBean(setup)

cpu_brand      = agent.si.get_cpu_brand(   )
cpu_cores      = agent.si.get_cpu_cores(   )
ram_total_Gb   = agent.si.get_ram_total(   )
gpu_brand      = agent.si.get_gpu_brand(   )
v_ram_total_Gb = agent.si.get_v_ram_total( )

ram_used_Gb    = [agent.si.get_ram_used()   ]
v_ram_used_Gb  = [agent.si.get_v_ram_used() ]
time_s         = [0.0                       ]
start_time     = time.time()

ram_label      = f"CPU: {cpu_brand}, {cpu_cores} Cores, RAM Total: {ram_total_Gb:6.2f} Gb, RAM Used: {agent.si.get_ram_used():6.2f} Gb, RAM Free: {agent.si.get_ram_free():6.2f} Gb"
v_ram_label    = f"GPU: {gpu_brand}, VRAM Total: {v_ram_total_Gb:6.2f} Gb, VRAM Used: {agent.si.get_v_ram_used():6.2f} Gb, VRAM Free: {agent.si.get_v_ram_free():6.2f} Gb"


# Define the function to be called when the Run button is pressed
def run_action(action_type, action_input):
    """ Run the action """
    output = ''.join(agent.agent_action(action_type, [action_input]))
    #print(f"LEN OUTPUT: {len(output)}")
    return output


def update_ram():
    """ Update the ram plot """
    #print(f"update_ram() called, elapsed: {time.time() - start_time:6.2f} s. ram_used_Gb: {agent.si.get_ram_used():6.2f} Gb.")
    ram_used_Gb.append(  agent.si.get_ram_used()   )
    v_ram_used_Gb.append(agent.si.get_v_ram_used() )
    time_s.append(       time.time() - start_time  )

    df         = pd.DataFrame({'time_s': time_s, 'ram_used_Gb': ram_used_Gb, 'v_ram_used_Gb': v_ram_used_Gb})
    update_ram = gr.LinePlot.update(
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
    update_v_ram = gr.LinePlot.update(
                        value  = df, 
                        title  = v_ram_label, 
                        x      = 'time_s', 
                        y      = 'v_ram_used_Gb',  
                        y_lim  = [0.0, v_ram_total_Gb],
                        height =  250, 
                        width  = 1000)

    return update_v_ram


# Define the Gradio display
with gr.Blocks(title="Agent Bean Interface") as iface:
    action_type          = gr.components.Dropdown(choices = agent.aa.get_available_actions(), label = "Action Type"   ) 
    action_input         = gr.components.Textbox( lines   = 5                               , label = "Action Input"  )
    run_button           = gr.Button(                                                         label = "Run"           )
    text_output          = gr.components.Textbox( lines   = 20,  autoscroll = True          , label = "Output Text"   )
    ram_plt              = gr.components.LinePlot(show_label=False)
    v_ram_plt            = gr.components.LinePlot(show_label=False)

    run_button.click( run_action, [action_type, action_input], outputs = text_output)

    dep_ram              = iface.load(update_ram  , None, ram_plt  , every=1)
    dep_v_ram            = iface.load(update_v_ram, None, v_ram_plt, every=1)

# Launch the interface
iface.queue().launch(share=False)
