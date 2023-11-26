# AI Agent Bean
Agent Bean doese not yet have the competencies of 007 it is more at Mr Mean level. It is AI agent that is designed to interact with users and perform various tasks. The initial focus is on coding but there are no limitations to other activities. 

It has been designed to be able to run local llm models like Llama2 and Mistral that are available on the huggingface portal as well as interfacing with the openAI API to be able to run GPT3.4 or GPT4. So you can run the model fully localy if you have the a GPU with enough VRAM or use OpenAI API you can even mix booths if it make sense for you. As you can associate different models to different actions.

It has a memory management system that can instantiate / dinstantiate models dependings on the needs of the task (it is still a bit rough )

It has been designed as a library that can be integrated into projects but there is as well a gradio interface to be able to test it directly.

Some of the code has been developped with the help of aider: https://github.com/paul-gauthier/aider (at the beginning while the code was less than 8k tokens)

## Curent capabilities:
The agent Bean is currently capable of the following:
- **Manage models**: 
   - Manage the models and ability to instantiate / deinstantiate models, 
   - Each tasks, can have their own model so you can use the best model for each task. 
   - Ability to use any transformers models, WizardCoder, Llama2, Mistral, dolphin-2.1-mistral-7b
   - Ability to use OpenAI: GPT-4, gpt-3.5-turbo, gpt-3.5-turbo-16k 
- **Agents**: have access to the following functions: 
   - **Load**: files from the following format: text, json, PDF 
   - **Search**: Search on the internet for informations. 
- **Action Execution**: Agent Bean can perform various actions. The actions are defined in the setings file and can be easily extended. curently implemented actions:
   - **free**: freetext query
   - **summarize**: generate a summary of the provided text
   - **search**: perform a search on the net
   - **split**: split a task executable actions 
   - **code**: generate code 
   - **code_quality**: analyse the quality of the code (not yet working well)
   - **project_requirements**: generate requirements
   - **team_manager**: experimental agent to bheave as a team manager, to see how a llm could bheave writing speaches for a team manager.


The agent is continuously evolving, with new capabilities being added regularly.


## Current limitations:
- When modifying the settings fil whlie the model is running, it needs to be re-uploaded for the changes to take effects
- It is not yet able to write to files
- Load on 4 or 8 bit loading works except for GPTQ or GGML quantized models!? (they use a different proces for loading that needs to be implemented)
- Imprecise memory estimation for models
- Tested only on linux up to now but should work on windows and Mac as well


## Using local generation with transformers models:
- You need at least 16GB of video RAM to run 15B models in 4bits to be able to have correct coding and splitting result
- With 8GB V RAM you can run the 7B models in 4Bits there you need to select the realy best ones like 'deepseek-coder-6.7b-instruct' as codding is demanding
- You may use a computer with less V RAM for testing purpose with 3B or 1.5B models like deepseek-coder-1.3b-instruct but do not expect usefull results.


## Backlog and Work in Progress (WIP):
- **Improve Model managment** Model memory estimation depending on model quantisation
- **Encapsulate model input outputs in objects**: this will allow easier text manipulation, parsing and treatments in a single place 
- **Ability to perform action on files** to have an actual effect on code
- **improve transformers loading**: for 8 and 4 bits quantisation with GPTQ and GGML models
- **Actions backlog**: a backlog of actions to be executed
- **Ability to execute code in a sealed container**: to provide direct feedback to the agent
- **Context Management**: Agent Bean maintains a context of the conversation, which is used to generate relevant responses. It can add new elements to the context, clear the context, and manage the context length to ensure it stays within a specified token limit.
- **Model and Vectorstore Instantiation**: Agent Bean can instantiate different models and vectorstores based on the provided setup. And be ablee to load documents in the vector store
- **Improve code quality verification action**: code quality checker is not yet working well (need to improve the prompt)
- **Task looping** ability to loop on repetitive tasks
- **Classifier action** to classify inputs
- **Implement LLAVA model** to interact with images/charts 


## Recent improvments:
- **Added File loader**: to be able to load text, json, or PDF files
- **Added settings file selection and loading**: to be able to easyly change the settings file
- **added a team_manager agent**: not totaly related to the rest but it is an experiment 
- **added requirements**: generate requirements
- **Make actions more generic** to allow beter variations from congig
- **Improve Model managment** Removed memory leak and improved model managment


## Useage:

Clone this repository:

`git clone https://github.com/kolergy/agent_bean.git`

Create an environement:

`conda create -n AB311 python=3.11`

Activate the environement:

`conda activate AB311`

Install the requirements:

`pip install -U -r requirements.txt`

If you intent to use the openAI models you have to rename the template.env file as follows:

`cp template.env .env`

Then edit this .env file to insert your openAI API key 

Once edited you need to set the environement with:

`sourrce .env`

To test the code you can run the interface:

`python agent_bean_interface.py`
