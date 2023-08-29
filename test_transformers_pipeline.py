import unittest
from transformers_pipeline import TfPipeline
from system_info import SystemInfo

class TestTfPipeline(unittest.TestCase):
    def setUp(self):
        # Load the settings json file and create a TfPipeline object
        with open('settings.json') as f:
            setup  = json.load(f)
        self.system_info = SystemInfo()
        self.pipeline = TfPipeline(setup['AgentBean_settings'], self.system_info)

    def test_instantiate_pipeline(self):
        # Test the instantiate_pipeline method
        self.pipeline.instantiate_pipeline()
        self.assertIsNotNone(self.pipeline.pipeline)

if __name__ == '__main__':
    unittest.main()
