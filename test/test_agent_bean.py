import unittest
import json    
import wikipedia  as     wiki
from   agent_bean.agent_bean import AgentBean


class TestAgentBean(unittest.TestCase):
    def setUp(self):
        # Load the settings json file and create a AgentBean object
        with open('settings_opai.json') as f:
            setup  = json.load(f)
        self.agent = AgentBean(setup['AgentBean_settings'])


    def tearDown(self):
        # clear the context
        self.agent.clear_context()


    def test_instantiate_model(self):
        # Test the instantiate_model method
        self.assertIsNotNone(self.agent.model)


    def test_instantiate_vectorstore(self):
        # Test the instantiate_vectorstore method
        self.assertIsNotNone(self.agent.v_db)


    def test_agent_action(self):
        # Test the agent_action method
        action_type    = 'summarize'
        wikipedia_page = wiki.page("Python (programming language)")
        input          = wikipedia_page.content
        if len(input) > self.agent.setup['model']['max_tokens']:
            input = input[:self.agent.setup['model']['max_tokens']]
        inputs         = [input]
        response       = self.agent.agent_action(action_type, inputs)
        print(f"TST ACTION: {response}")
        self.assertTrue(len(response) > 10)


    def test_manage_context_length(self):
        # Test the manage_context_length method
        wikipedia_page1   = wiki.page("Python (programming language)")
        wikipedia_page2   = wiki.page("Neural network")
        self.agent.add_context([wikipedia_page1.content, wikipedia_page2.content])
        self.agent.manage_context_length()
        print(f"TST CTX: context length: {sum(len(c) for c in self.agent._context)}")
        self.assertTrue(sum(len(c) for c in self.agent._context) < 0.8 * self.agent.setup['model']['max_tokens'])


    def test_load_document(self):
        # Test the load_document method
        files_path = ['README.md']
        self.agent.load_document(files_path)
        self.assertTrue(self.agent.v_db.index.ntotal > 0)


if __name__ == '__main__':
    unittest.main()
