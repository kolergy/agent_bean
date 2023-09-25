import json
import time

import gradio                as gr

from   agent_bean.agent_bean import AgentBean

#settings_file = 'settings_opai.json'
settings_file = 'settings_trans.json'

# Load the settings json file and create a AgentBean object
with open(settings_file) as f:
    setup = json.load(f)
agent = AgentBean(setup)


# Define the function to be called when the Run button is pressed
def run_action(action_type, action_input):
    output = ''.join(agent.agent_action(action_type, [action_input]))
    print(f"LEN OUTPUT: {len(output)}")
    return output


# Define the function to get system information
def get_system_info():
    cpu_brand      = agent.system_info.get_cpu_brand(      )
    cpu_cores      = agent.system_info.get_cpu_cores(      )
    ram_total      = agent.system_info.get_ram_total(      )
    ram_used       = agent.system_info.get_ram_used(       )
    gpu_brand      = agent.system_info.get_gpu_brand(      )
    vram_total     = agent.system_info.get_vram_total(     )
    vram_used      = agent.system_info.get_vram_used(      )
    vram_available = agent.system_info.get_vram_available( )
    return f"CPU Brand: {cpu_brand}\nCPU Cores: {cpu_cores}\nRAM Total: {ram_total} GB\nRAM Used: {ram_used} GB\nGPU Brand: {gpu_brand}\nVRAM Total: {vram_total} GB\nVRAM Used: {vram_used} GB\nVRAM Available: {vram_available} GB"


# Define the Gradio display
with gr.Blocks(title="Agent Bean Interface") as iface:
    action_type          = gr.components.Dropdown(choices = agent.aa.get_available_actions(), label = "Action Type"  ) 
    action_input         = gr.components.Textbox( lines   = 5                               , label = "Action Input" )
    run_button           = gr.Button(                                                         label = "Run"          )
    text_output          = gr.components.Textbox( lines   = 20, autoscroll = True           , label = "Output Text"  )

    run_button.click( run_action, [action_type, action_input], outputs = text_output)

# Launch the interface
iface.launch(share=True)
