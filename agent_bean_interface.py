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

# Define the Gradio interface
iface = gr.Interface(
    fn          = run_action, 
    inputs      = [
        gr.inputs.Dropdown(choices=["summarize", "search"], label="Action Type"),
        gr.inputs.Textbox(lines=5, label="Action Input")
    ], 
    outputs     = "text",
    title       = "Agent Bean Interface",
    description = "Select an action and provide the input for the action. Then press Run to execute the action."
)

# Launch the interface
iface.launch()
