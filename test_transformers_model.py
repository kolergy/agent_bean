
import json  
import unittest

from   transformers_model import TfModel
from   system_info           import SystemInfo

class TestTfPipeline(unittest.TestCase):
    def setUp(self):
        # Load the settings json file and create a TfPipeline object
        with open('settings.json') as f:
            setup  = json.load(f)
        self.system_info = SystemInfo()
        self.TF_pipeline = TfModel(setup['AgentBean_settings'], self.system_info)

    def test_instantiate_pipeline(self):
        # Test the instantiate_pipeline method
        self.assertIsNotNone(self.TF_pipeline.pipeline)

if __name__ == '__main__':
    unittest.main()
