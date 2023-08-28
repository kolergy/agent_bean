# Agent Bean
Agent Bean is a Language Model (LLM) agent that is designed to interact with users and perform various tasks. 
It is being developped with the help of aider: https://github.com/paul-gauthier/aider

It is currently capable of the following:

1. **Context Management**: Agent Bean maintains a context of the conversation, which is used to generate relevant responses. It can add new elements to the context, clear the context, and manage the context length to ensure it stays within a specified token limit.

2. **Action Execution**: Agent Bean can perform various actions such as summarizing text and searching the internet. The actions are defined in the `AgentAction` class and can be easily extended.

3. **Document Loading**: Agent Bean can load a set of documents into a vectorstore for later use.

4. **Model and Vectorstore Instantiation**: Agent Bean can instantiate different models and vectorstores based on the provided setup.

The agent is continuously evolving, with new capabilities being added regularly.
