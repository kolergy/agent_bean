import json

import gradio as gr

from agent_bean import AgentBean

# Load the settings json file and create a AgentBean object
with open('settings.json') as f:
    setup = json.load(f)
agent = AgentBean(setup['AgentBean_settings'])

# Define the function to be called when the Run button is pressed
def run_action(action_type, action_input):
    return agent.agent_action(action_type, [action_input])

# Define the function to be called when the Load Document button is pressed
def load_document(document_path):
    agent.load_document([document_path])
    return "Document loaded successfully."

# Define the Gradio display
with gr.Blocks(title="Agent Bean Interface") as iface:
    action_type          = gr.inputs.Dropdown(choices = ["summarize", "search"], label = "Action Type" ) 
    action_input         = gr.inputs.Textbox( lines   = 5,                       label = "Action Input")
    run_button           = gr.Button(  label   = "Run"          )
    document_path        = gr.inputs.File(    label   = "Document Path")
    load_document_button = gr.Button(  label   = "Load Document")
    text                 = gr.outputs.Textbox(label   = "Output Text"  )

    run_button.click(run_action, [action_type, action_input], outputs = text)
    load_document_button.click(load_document, [document_path], outputs = text)
   


# Launch the interface
iface.launch()
