
import os
import json
import torch

from   transformers_model                import TfModel
from   langchain.chat_models             import ChatOpenAI
from   langchain.embeddings.openai       import OpenAIEmbeddings
from   system_info                       import SystemInfo


class ModelsManager():
    """This class is used to manage the models and their ressources useage"""
    def __init__(self, setup) -> None:
        self.setup             = setup
        self.system_info       = SystemInfo()
        self.debug             = setup['debug']
        self.active_models     = {}
        self.active_embeddings = {}
        if os.path.exists(self.setup["known_models_file_name"]):
            with open(self.setup["known_models_file_name"]) as f:
                self.known_models  = json.load(f)
        else:
            self.known_models      = {}
        self.test_models_ressources_reqs()
    

    def model_need(self, model_name:str) -> bool:
        """check memory resources and instantiate a nneded model if not already instantiated, may remove other instantiated models if needed"""
        if model_name not in self.active_models:
            if self.debug:
                print(f"Model {model_name} not yet instantiated, instantiating it now")
            if self.manage_mem_resources(model_name):
                self.instantiate_model(model_name)
                if model_name in self.active_models:
                    return True
                else:
                    print(f"ERROR: Model {model_name} not instantion failed")
                    return False
            else:
                print(f"ERROR: Model {model_name} not enough resources for instantion")
                return False
        else:
            if self.debug:
                print(f"Model {model_name} already instantiated")
            return True
        
    def predict(self, model_name:str, prompt:str, max_tokens:int, temperature:float, top_p:float, frequency_penalty:float, presence_penalty:float, stop:list) -> str:
        """predict using a model"""
        if self.model_need(model_name):
            return self.active_models[model_name].predict(prompt, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop)
        else:
            return None

    def manage_mem_resources(self, model_name:str) -> bool:
        """check model memory need va available resources may remove other instantiated models if needed"""
        if self.debug:
            print(f"Checking memory resources for model {model_name}")
            self.system_info.print_GPU_info()
        if model_name not in self.known_models:
            print(f"ERROR: Model {model_name} not in known_models")
            return False
        else:
            if self.debug:
                print(f"Model {model_name} in known_models checking memory resources")
                print(f"      RAM need: {self.known_models[model_name]['ram_need'  ]} Gb, system available RAM: {self.system_info.get_ram_free()  } Gb")
                print(f"Video RAM need: {self.known_models[model_name]['v_ram_need']} Gb,  GPU available V RAM: {self.system_info.get_v_ram_free()} Gb")
            if self.known_models[model_name]['ram_need'  ] > self.system_info.get_ram_free()  or self.known_models[model_name]['v_ram_need'] > self.system_info.get_v_ram_free():
                return self.free_resources(self.known_models[model_name]['ram_need'  ], self.known_models[model_name]['v_ram_need'])
            else:
                return True
            

    def free_resources(self, required_free_ram_gb:float, required_free_v_ram_gb:float) -> bool:
        """free resources to meet the required free resources"""
        ram_contrib_ratio     = {}
        v_ram_contrib_ratio   = {}
        current_free_ram_gb   = self.system_info.get_ram_free()
        current_free_v_ram_gb = self.system_info.get_v_ram_free()
        ram_need_gb           = required_free_ram_gb   - current_free_ram_gb
        v_ram_need_gb         = required_free_v_ram_gb - current_free_v_ram_gb
        if len(self.active_models) > 0:
            for model_name in self.active_models.keys():
                if ram_need_gb > 0:
                    ram_contrib_ratio[model_name] = self.known_models[model_name]['ram_need'  ] / ram_need_gb
                    if ram_contrib_ratio[model_name] > 1:
                        FLAG_RAM_Ok = True
                    else:
                        FLAG_RAM_Ok = False
                else:
                    FLAG_RAM_Ok = True
                if v_ram_need_gb > 0:
                    v_ram_contrib_ratio[model_name] = self.known_models[model_name]['v_ram_need'] / v_ram_need_gb
                    if v_ram_contrib_ratio[model_name] > 1:
                        FLAG_V_RAM_Ok = True
                    else:
                        FLAG_V_RAM_Ok = False
                else:
                    FLAG_V_RAM_Ok = True

                if FLAG_RAM_Ok and FLAG_V_RAM_Ok:
                    self.deinstantiate_model(model_name)
                    current_free_ram_gb   = self.system_info.get_ram_free()
                    current_free_v_ram_gb = self.system_info.get_v_ram_free()
                    if self.debug:
                        print(f"Model {model_name} deinstantiated")
                    if current_free_ram_gb >= required_free_ram_gb and current_free_v_ram_gb >= required_free_v_ram_gb:
                        return True
                    else:
                        print(f"ERROR: Model {model_name} deinstantiation did not recover enough resources: free RAM: {current_free_ram_gb} Gb, free V RAM: {current_free_v_ram_gb} Gb")
                        return False
            if current_free_ram_gb >= required_free_ram_gb and current_free_v_ram_gb >= required_free_v_ram_gb:
                return True
            else:
                print(f"ERROR: Model {model_name} deinstantiation full loop did not recover enough resources: free RAM: {current_free_ram_gb} Gb, free V RAM: {current_free_v_ram_gb} Gb")
                return False            
        else:   
            if self.debug:
                print(f"No active model -> model to deinstantiate")
            return False


    def test_models_ressources_reqs(self) -> None:
        """Test the models ressources requirements and update the known_models dict accordingly"""
        if self.debug:
            print(f"Testing models memory ressources requirements")
            self.system_info.print_GPU_info()
        for model_name in self.setup["models_list"].keys():
            if self.setup["models_list"][model_name]["model_id"] not in self.known_models.keys():
                self.known_models[model_name]["model_id"] = {}
                if self.debug:
                    print(f"Model {model_name}: is not yet known Testing model memory ressources requirements")
                    self.system_info.print_GPU_info()
                    
                if self.setup["models_list"][model_name]["model_type"] == "transformers":
                    ram_b4   = self.system_info.get_ram_free()
                    v_ram_b4 = self.system_info.get_v_ram_free()
                    self.instantiate_model(model_name)
                    
                    delta_ram_gb   = max(0, ram_b4   - self.system_info.get_ram_free()  ) # min value is 0
                    delta_v_ram_gb = max(0, v_ram_b4 - self.system_info.get_v_ram_free()) # to avoid noise on the unused ram
                    self.known_models[model_name]["system_ram_gb"] = delta_ram_gb
                    self.known_models[model_name]["GPU_ram_gb"   ] = delta_v_ram_gb
                    if self.debug:
                        print(f"Model {model_name} instantiated using: {delta_ram_gb} Gb of system RAM and: {delta_v_ram_gb} Gb of V RAM on the GPU")
                        self.system_info.print_GPU_info()

                    self.deinstantiate_model(model_name)

                elif self.setup["models_list"][model_name]["model_type"] == "openAI":
                    self.known_models[model_name]["v_ram_need"] = 0.0
                    self.known_models[model_name]["ram_need"  ] = 0.0

                else:
                    print(f"ERROR: Unknown model type: {self.setup['models_list'][model_name]['model_type']}")

        with open(self.setup["known_models_file_name"], 'w') as f:  # store the known models dict to a file to avoid doing it again
            json.dump(self.known_models, f, indent=4)

        if self.debug:
            print(f"Models memory ressources requirements test complete")
            self.system_info.print_GPU_info()


    def instantiate_model(self, model_name:str) -> None:
        """instantiate the model defined in the set-up by adding it to the active model list and creating the corresponding embeddings"""
        if self.setup['model']['model_type'] == "openAI":
            api_key                        = os.getenv('OPENAI_API_KEY')
            self.active_models[model_name] = ChatOpenAI(openai_api_key=api_key, model_name=self.setup['model']['model_id'])
            self.embeddings[model_name]    = OpenAIEmbeddings(openai_api_key=api_key)

        elif self.setup['model']['model_type'] == "transformers":
            if self.debug:
                print(f"GPU state before model instantiation: {torch.cuda.is_available()}")
                self.system_info.print_GPU_info()

            self.active_models[model_name] = TfModel(self.setup, self.system_info)
            self.embeddings[model_name]    = self.active_models[model_name].embeddings
            if self.debug:
                print(f"GPU state after model instantiation: {torch.cuda.is_available()}")
                self.system_info.print_GPU_info()
    

    def deinstantiate_model(self, model_name:str) -> None: 
        """deinstantiate the model provided in the argument end ensure proper deletion of all the model ressources"""
        if self.debug:
            print(f"GPU state before model deinstantiation: {torch.cuda.is_available()}")
            self.system_info.print_GPU_info()
        self.embeddings[model_name].__del__()
        self.active_models[model_name].free()
        self.active_models[model_name].cuda()

        self.embeddings.pop(model_name)
        self.active_models.pop(model_name)

        if torch.cuda.is_available():
            print("-m--Emptying CUDA cache----")
            torch.cuda.empty_cache()

        if self.debug:
            print(f"GPU state after model deinstantiation: {torch.cuda.is_available()}")
            self.system_info.print_GPU_info()
        
