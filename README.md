# AI Agent Bean
Agent Bean does not yet have the competencies of 007; it is more at Mr Bean's level. It is an AI agent that is designed to interact with users and perform various tasks. The initial focus was on coding, but some experimentation on team management are surprisingly good; there are no limitations to other activities. 

It has been designed to be able to run local llm models like Llama2 and Mistral, zephyr-7b-beta, Orca-2, WizardCoder, etc... that are available on the huggingface portal in totally private mode with out any API but it can as well be interfaced with the openAI API to be able to run GPT3.4 or GPT4 and the new GPT4 turbo If you want. And just NOW you can use the Models from Mistral API. So you can run the model fully localy if you have the a GPU with enough VRAM or use OpenAI API you can even mix booths if it make sense for you. As you can associate different models to different actions.

It has a memory management system that can instantiate / dinstantiate models dependings on the needs of the task (it is still a bit rough )
It has been designed as a library that can be integrated into projects but there is as well a gradio interface to be able to test it directly.

Some of the code has been developped with the help of aider: https://github.com/paul-gauthier/aider (at the beginning while the code was less than 8k tokens)

## Curent capabilities:
The Agent Bean is currently capable of the following:

- **Action Execution**: Agent Bean can perform various actions. The actions are defined in the settings file and can be easily extended. Currently implemented actions:
   - **free**: Free text query
   - **summarize**: Generate a summary of the provided text
   - **search**: Perform a search on the net
   - **split**: Split a task into executable actions 
   - **code**: Generate code 
   - **code_quality**: analyse the quality of the code (not yet working well)
   - **project_requirements**: generate requirements
   - **team_manager_speach**: experimental agent to bheave as a team manager, to see how a llm could bheave writing speaches for a team manager.
   - **team_manager_121**: experimental agent to bheave as a team manager, to see how a llm could bheave having a one to one speach.
   - **sports_coach**: Coach to help with the preparation of High level athletes
   - **ideation_coach**: experimental agent to behave as an innovation coach
   - **Q_and_A_as_human**: Questions and answers with Arthur who tries to bheaves as a human
   - **Q_and_A_generator**: Generate Questions and answers from the provided text
   - **meal_planner**: Plan meals depending of your tastes
   - **acoustic_agent**: agent to help you to solve issues with your acoustic system
   - **cybersecurity_agent**: agent to help you with cybersecurity issues
   - **comunication_agent**: Agent to help with the fablab communication in French

- **Actions**: have access to the following functions: 
   - **Load**: files from the following format: text, json, PDF 
   - **Search**: Search on the internet for informations. 

- **Manage models**: 
   - Manage the models and ability to instantiate / deinstantiate models, 
   - Each tasks, can have their own model so you can use the best model for each task. 
   - Ability to use any transformers models, WizardCoder, Llama2, Mistral, dolphin-2.1-mistral-7b
   - Ability to use OpenAI: GPT-4, gpt-3.5-turbo, gpt-3.5-turbo-16k 

   
The agent is continuously evolving, with new capabilities being added regularly.


## Current limitations:
- No check on contect lenght for now
- It is not yet able to write to files.
- Load on 4 or 8 bit loading works except for GPTQ or GGML quantized models! (they use a different proces for loading that needs to be implemented)
- When modifying the settings file whlie the model is running, it needs to be re-uploaded for the changes to take effects
- Tested only on linux up to now but should work on windows and Mac as well


## Using local generation with transformers models:
- You need at least 16GB of video RAM to run 15B models in 4bits to be able to have correct coding and splitting result
- With 8GB V RAM you can run the 7B models in 4Bits there you need to select the realy best ones like 'deepseek-coder-6.7b-instruct' as codding is demanding
- You may use a computer with less V RAM for testing purpose with 3B or 1.5B models like deepseek-coder-1.3b-instruct but do not expect usefull results.


## Set-up:

Clone this repository:

`git clone https://github.com/kolergy/agent_bean.git`

Create an environement:

`conda create -n AB311 python=3.11`

Activate the environement:

`conda activate AB311`

Install the requirements:

`pip install -U -r requirements.txt`

If you intent to use the openAI or Mistral models via their API you have to rename the template.env file as follows:

`cp template.env .env`

Then edit this .env file to insert your openAI or Mistral API key 

Once edited you need to set the environement with:

`sourrce .env`

To test the code you can run the interface:

`python agent_bean_interface.py`



## Recent improvments:

- [x] **Context Management**: Agent Bean maintains a context of the conversation, which is used to generate relevant responses. elements are add to the context as exchanges progresses, clear the context, 
- [x] **Added transformers_local setting**: gives you the possibility to run local only with transformers you have pre-loaded
- [x] **Added batch agent**: run the action list without the interface
- [x] **Added action list**: ability to run an action list
- [x] **Encapsulate model input outputs in objects**: this will allow easier text manipulation, parsing, and treatments in a single place 
- [x] **Added the console log**: you will have the console log directly in the gradio app so you se what is going on.
- [x] **Added Mistral API**: you can now use the models from the Mistral API!
- [x] **Added ideation_coach**: to guide innovators with refining their innovations.
- [x] **Added File loader**: to be able to load text, JSON, or PDF files
- [x] **Added settings file selection and loading**: to be able to easyly change the settings file
- [x] **added a team_manager agent**: not totaly related to the rest but it is an experiment 
- [x] **added requirements**: generate requirements
- [x] **Make actions more generic** to allow beter variations from congig
- [x] **Improve Model managment** Removed memory leak and improved model managment
- [x] **Refactor update_num_tokens**: Updated the method to calculate the number of tokens using the model tokenizer for more accurate token counting.


## Backlog and Work in Progress (WIP): 

- [ ] **Display API key availability on interface** to see if mistral or OpenAI API keys are present in env
- [ ] **Improve Model management** Model memory estimation depending on model quantization
- [ ] **Refactor update_num_tokens**: Refactor the method to calculate the number of tokens using the model tokenizer.
- [ ] **Ability to perform action on files** to have an actual effect on code
- [ ] **improve transformers loading**: GPTQ and GGML models
- [ ] **Actions backlog**: a backlog of actions to be executed
- [ ] **Ability to execute code in a sealed container**: to provide direct feedback to the agent
- [ ] **Improve Context Management**: manage context lenght
- [ ] **Improve Context Management**: clear the system delimiters from the context
- [ ] **Model and Vectorstore Instantiation**: Agent Bean can instantiate different models and vectorstores based on the provided setup. And be ablee to load documents in the vector store
- [ ] **Improve code quality verification action**: code quality checker is not yet working well (needs to improve the prompt).
- [ ] **Task looping** ability to loop on repetitive tasks
- [ ] **Classifier action** to classify inputs
- [ ] **Implement LLAVA model** to interact with images/charts 


