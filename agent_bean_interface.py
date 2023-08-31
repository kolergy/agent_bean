import json
import time

import gradio as gr

from agent_bean import AgentBean

# Load the settings json file and create a AgentBean object
with open('settings_opai.json') as f:
    setup = json.load(f)
agent = AgentBean(setup['AgentBean_settings'])

# Define the function to be called when the Run button is pressed
def run_action(action_type, action_input):
    return agent.agent_action(action_type, [action_input])

# Define the function to be called when the Load Document button is pressed
def load_document(document_path):
    agent.load_document([document_path])
    return "Document loaded successfully."

# Define the function to get system information
def get_system_info():
    cpu_brand      = agent.system_info.get_cpu_brand()
    cpu_cores      = agent.system_info.get_cpu_cores()
    ram_total      = agent.system_info.get_ram_total()
    ram_used       = agent.system_info.get_ram_used()
    gpu_brand      = agent.system_info.get_gpu_brand()
    vram_total     = agent.system_info.get_vram_total()
    vram_used      = agent.system_info.get_vram_used()
    vram_available = agent.system_info.get_vram_available()
    return f"CPU Brand: {cpu_brand}\nCPU Cores: {cpu_cores}\nRAM Total: {ram_total} GB\nRAM Used: {ram_used} GB\nGPU Brand: {gpu_brand}\nVRAM Total: {vram_total} GB\nVRAM Used: {vram_used} GB\nVRAM Available: {vram_available} GB"

# Define the Gradio display
with gr.Blocks(title="Agent Bean Interface") as iface:
    action_type          = gr.components.Dropdown(choices = ["summarize", "search"], label = "Action Type" ) 
    action_input         = gr.components.Textbox( lines   = 5,                       label = "Action Input")
    run_button           = gr.Button(             label   = "Run"          )
    document_path        = gr.components.File(    label   = "Document Path")
    load_document_button = gr.Button(             label   = "Load Document")
    text                 = gr.components.Textbox( label   = "Output Text"  )
    system_info          = gr.components.Textbox( label   = "System Info"  )

    run_button.click(          run_action   , [action_type, action_input], outputs = text)
    load_document_button.click(load_document, [document_path]            , outputs = text)
    


# Launch the interface
iface.launch(share=True)
