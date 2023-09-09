# Agent Bean
Agent Bean is a Language Model (LLM) agent that is designed to interact with users and perform various tasks. 
It is being developped with the help of aider: https://github.com/paul-gauthier/aider
the aim is to have an agent capable to be run fully localy, it is still possible to usie openAI API


## Curent capabilities:
The agent Bean is currently capable of the following:
- **Manage models**: manage the models and ability to instantiate / deinstantiate models, ability to use any transformers models 
- **Action Execution**: Agent Bean can perform various actions. The actions are defined in the `AgentAction` class and can be easily extended. curently implemented actions:
  -  **summarize**: generate a summary of the provided text
  -  **search**: perform a search on the net
  -  **split**: split a task executable actions (not yet working well)
  -  **code**: generate code (not yet working well)
  -  **code_quality**: analyse the quality of the code (not yet working well)

The agent is continuously evolving, with new capabilities being added regularly.

## WIP: Work in Progress:
- **improve transformers loading**: for 8 and 4 bits quantisation
- **Actions backlog**: a backlog of actions to be executed
- **Ability to execute code in a sealed container**: to provide direct feedback to the agent
- **Context Management**: Agent Bean maintains a context of the conversation, which is used to generate relevant responses. It can add new elements to the context, clear the context, and manage the context length to ensure it stays within a specified token limit.
- **Document Loading**: Agent Bean can load a set of documents into a vectorstore for later use.
- **Model and Vectorstore Instantiation**: Agent Bean can instantiate different models and vectorstores based on the provided setup.

## Useage:

Create an environement:

`conda create -n AB311 python=3.11`

Activate the environement:

`conda activate AB311`

Install the requirements:

`pip install -U -r requirements.txt`

Run the agent bean interface:

`python agent_bean_interface.py`


If you intent to use the openAI models you have to rename the template.env file as follows:
`cp template.env .env`

Then edit this .env file to insert your openAI API key
