import json

import gradio       as     gr

from   agent_bean   import AgentBean


# Load the settings json file and create a AgentBean object
with open('settings.json') as f:
    setup  = json.load(f)
agent = AgentBean(setup['AgentBean_settings'])

# Define the function to be called when the Run button is pressed
def run_action(action_type, action_input):
    return agent.agent_action(action_type, [action_input])

# Define the function to be called when the Load Document button is pressed
def load_document(document_path):
    agent.load_document([document_path])
    return "Document loaded successfully."

# Define the Gradio interface
iface = gr.Interface(
    fn   = {
        "run_action": run_action,
        "load_document": load_document
    },
    inputs      = [
        gr.components.Dropdown(choices=["summarize", "search"], label="Action Type"),
        gr.components.Textbox(lines=5, label="Action Input"),
        gr.components.Textbox(label="Document Path")
    ],
    outputs     = "text",
    title       = "Agent Bean Interface",
    description = "Select an action and provide the input for the action. Then press Run to execute the action. You can also load a document into the vectorstore by providing its path and pressing Load Document."
)

# Launch the interface
iface.launch()
