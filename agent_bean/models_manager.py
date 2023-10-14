
import os
import sys
import gc
import json
import torch
import tiktoken

from   langchain.chat_models             import ChatOpenAI
from   langchain.embeddings.openai       import OpenAIEmbeddings
from   agent_bean.system_info            import SystemInfo
from   agent_bean.transformers_model     import TfModel, TransformersEmbeddings


class ModelsManager():
    """This class is used to manage the models and their ressources useage"""
    def __init__(self, setup: dict, si: SystemInfo) -> None:

        self.setup_update(setup)
        self.setup                    = setup
        self.si                       = si
        self.debug                    = setup['debug']
        self.active_models            = {}
        self.active_embeddings        = {}
        self.known_models             = {}
        self.openai_params_list       = ["temperature", "max_tokens"          ]
        self.transformers_params_list = ["temperature", "max_tokens", "stop", "presence_penalty", "frequency_penalty", "top_p" ]

        self.test_models_resources_reqs()
    

    def setup_update(self, setup: dict) -> None:
        """update the setup and deinstantiate models and embeddings if needed"""
        self.setup = setup
        self.debug = setup['debug']
        if os.path.exists(self.setup["known_models_file_name"]):
            with open(self.setup["known_models_file_name"]) as f:
                self.known_models  = json.load(f)
        if 'reload_models' in self.setup.keys():
            if self.setup['reload_models']:
                if len(self.active_models) > 0:
                    for m in self.active_models.keys():
                        self.deinstantiate_model(m)


    def model_need(self, model_name:str) -> bool:
        """if model not already instantiated check memory resources and instantiate a nneded model, may remove other instantiated models if needed"""
        if model_name not in self.active_models:
            if self.debug:
                print(f"Model {model_name} not yet instantiated, instantiating it now")
            if self.manage_mem_resources(model_name):
                self.instantiate_model(model_name)
                if model_name in self.active_models:
                    return True
                else:
                    print(f"ERROR: Model {model_name} instantion failed")
                    raise  RuntimeError(f"ERROR: Model {model_name} instantion failed")
            else:
                print(f"ERROR: Model {model_name} not enough resources for instantion")
                raise RuntimeError(f"ERROR: Model {model_name} not enough resources for instantion")
        else:
            if self.debug:
                print(f"Model {model_name} already instantiated")
            return True
        

    def set_model_params(self, model_name:str, params:dict) -> None:
        """set model parameters"""
        if self.model_need(model_name):
            for p in params.keys():
                #print(f" model type: {type(self.active_models[model_name])}, param: {p}, value: {params[p]}")
                #print(f" openAI type: {type(ChatOpenAI)}, transformers type: {type(TfModel)}")
                if isinstance(self.active_models[model_name],ChatOpenAI): 
                    if p in self.openai_params_list:
                        print(f"setting param: {p}, value: {params[p]}")
                        setattr(self.active_models[model_name], p, params[p])
                elif isinstance(self.active_models[model_name], TfModel):
                    if p in self.transformers_params_list:
                        setattr(self.active_models[model_name], p, params[p])
                else:
                    print(f"ERROR: Unknown model type: {type(self.active_models[model_name])}")
        else:
            print(f"ERROR: can not set params Model {model_name} could not be instantiated")


    def predict(self, model_name:str, prompt:str ) -> str:
        """predict using a model"""
        if self.model_need(model_name):
            res =  self.active_models[model_name].predict(prompt)
            print(f"predict result: {res}") 
            return [res]
        else:
            return None


    def decode(self, model_name:str, tokens:[float]) -> str:
        """decode using a model"""
        if self.model_need(model_name):
            return self.active_embeddings[model_name].decode(tokens)
        else:
            return None


    def get_embeddings(self, model_name:str, text:str) -> torch.tensor:
        """get embeddings using a model"""
        if self.model_need(model_name):
            return self.active_embeddings[model_name].encode(text)
            #return self.active_embeddings[model_name].embed_query(text)
        else:
            return None


    def manage_mem_resources(self, model_name:str) -> bool:
        """check model memory need vs available resources, may remove other instantiated models if needed"""
        k_model_id = TfModel.keyify_model_id(self.setup['models_list'][model_name]['model_id'])
        if self.debug:
            print(f"Checking memory resources for model {k_model_id}")
            self.si.print_GPU_info()
        if k_model_id not in self.known_models:
            print(f"WARNING: Model {k_model_id} not in known_models list, running blind de instanciating everything just in case") 
            keys = list(self.active_models.keys())  # avoid -> RuntimeError: dictionary changed size during iteration
            for m in keys:
                self.deinstantiate_model(m)
            return True
        else:
            if self.debug:
                print(f"Model {k_model_id} in known_models checking memory resources")
                print(f"      RAM need: {self.known_models[k_model_id]['system_ram_gb'  ]} Gb, system available RAM: {self.si.get_ram_free()  } Gb")
                print(f"Video RAM need: {self.known_models[k_model_id]['GPU_ram_gb']} Gb,  GPU available V RAM: {self.si.get_v_ram_free()} Gb")
            if self.known_models[k_model_id]['system_ram_gb'  ] > self.si.get_ram_free()  or self.known_models[k_model_id]['GPU_ram_gb'] > self.si.get_v_ram_free():
                reaming_ram_gb   = self.si.get_ram_total(  ) - self.known_models[k_model_id]['system_ram_gb']
                reaming_v_ram_gb = self.si.get_v_ram_total() - self.known_models[k_model_id]['GPU_ram_gb'   ]
                models_ram_use_gb   = []
                models_v_ram_use_gb = []
                models_value        = []
                models_names        = list(self.active_models.keys())  # avoid -> RuntimeError: dictionary changed size during iteration
                n                   = len(models_names)
                models_value        = [1]*n
                for m in models_names:
                    k_m_id = TfModel.keyify_model_id(self.setup['models_list'][m]['model_id'])
                    models_ram_use_gb.append(  self.known_models[k_m_id]['system_ram_gb'] )
                    models_v_ram_use_gb.append(self.known_models[k_m_id]['GPU_ram_gb'   ] )

                print(f"TEST KNAPSACK: \ninputs: reaming_ram_gb: {reaming_ram_gb},\nmodels_ram_use_gb: {models_ram_use_gb}, \nmodels_value: {models_value}, \nmodels_names: {models_names}, n: {n}")
                print(f"\ninputs: reaming_v_ram_gb: {reaming_v_ram_gb},\n models_v_ram_use_gb: {models_v_ram_use_gb}")

                value_ram  , keep_models_ram   = self.__models_knap_sack__(reaming_ram_gb  , models_ram_use_gb  , models_value, models_names, n) 
                value_v_ram, keep_models_v_ram = self.__models_knap_sack__(reaming_v_ram_gb, models_v_ram_use_gb, models_value, models_names, n)

                print(f"keep_models_ram: {keep_models_ram}")
                print(f"keep_models_v_ram: {keep_models_v_ram}")

                for m in models_names:
                    #if m not in keep_models_ram or m not in keep_models_v_ram:
                        self.deinstantiate_model(m)
                        if self.debug:
                            print(f"manage_mem_resources: Model {m} deinstantiated")

                if self.known_models[k_model_id]['system_ram_gb'  ] > self.si.get_ram_free()  or self.known_models[k_model_id]['GPU_ram_gb'] > self.si.get_v_ram_free():
                    print(f"ERROR: manage_mem_resources failed to free enough resources to instantiate model {k_model_id}")
                    print(f"      RAM need: {self.known_models[k_model_id]['system_ram_gb'  ]} Gb, system available RAM: {self.si.get_ram_free()  } Gb")
                    print(f"Video RAM need: {self.known_models[k_model_id]['GPU_ram_gb']} Gb,  GPU available V RAM: {self.si.get_v_ram_free()} Gb")
                    return False
                else:
                    return True
            else:
                return True
            

    def free_resources(self, required_free_ram_gb:float, required_free_v_ram_gb:float) -> bool:
        """free resources to meet the required free resources"""
        ram_contrib_ratio     = {}
        v_ram_contrib_ratio   = {}
        current_free_ram_gb   = self.si.get_ram_free()
        current_free_v_ram_gb = self.si.get_v_ram_free()
        ram_need_gb           = required_free_ram_gb   - current_free_ram_gb
        v_ram_need_gb         = required_free_v_ram_gb - current_free_v_ram_gb
        if len(self.active_models) > 0:
            print(f"Freeing resources List of active models: {self.active_models.keys()}")
            for model_name in self.active_models.keys():
                k_model_id = TfModel.keyify_model_id(self.setup['models_list'][model_name]['model_id'])
                if ram_need_gb > 0:
                    ram_contrib_ratio[model_name] = self.known_models[k_model_id]['system_ram_gb'  ] / ram_need_gb
                    if ram_contrib_ratio[model_name] > 1:
                        FLAG_RAM_Ok = True
                    else:
                        FLAG_RAM_Ok = False
                else:
                    FLAG_RAM_Ok = True
                if v_ram_need_gb > 0:
                    v_ram_contrib_ratio[model_name] = self.known_models[k_model_id]['GPU_ram_gb'] / v_ram_need_gb
                    if v_ram_contrib_ratio[model_name] > 1:
                        FLAG_V_RAM_Ok = True
                    else:
                        FLAG_V_RAM_Ok = False
                else:
                    FLAG_V_RAM_Ok = True

                if FLAG_RAM_Ok and FLAG_V_RAM_Ok:
                    self.deinstantiate_model(model_name)
                    current_free_ram_gb   = self.si.get_ram_free()
                    current_free_v_ram_gb = self.si.get_v_ram_free()
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



    def __models_knap_sack__(self, avail_mem_gb:float, models_mem_gb:[float],  models_value:[float], models_names:[str], n) -> (float, [str]):
        """Knapsack algorithm to find the list of best models to keep in memory """
        if n == 0 or avail_mem_gb == 0:   # Base Case
            return 0, []
    
        # If weight of the nth item is more than Knapsack of capacity avail_mem_gb,
        # then this item cannot be included in the optimal solution
        if (models_mem_gb[n-1] > avail_mem_gb):
            return self.__models_knap_sack__(avail_mem_gb, models_mem_gb, models_value, models_names, n-1)
    
        # return the maximum of two cases: (1) nth item included
        v1, keep_models_1 = self.__models_knap_sack__(avail_mem_gb-models_mem_gb[n-1], models_mem_gb, models_value, models_names, n-1)
        v1               += models_value[n-1]
        keep_models_1     = [models_names[n-1]] + keep_models_1
        
        # (2) not included
        v2, keep_models_2 = self.__models_knap_sack__(avail_mem_gb, models_mem_gb, models_value, models_names, n-1)
        
        return (v1, keep_models_1) if v1 >= v2 else (v2, keep_models_2)
        



    def test_models_resources_reqs(self) -> None:
        """Test the models ressources requirements and update the known_models dict accordingly"""
        if self.debug:
            print(f"Testing models memory ressources requirements")
            self.si.print_GPU_info()
        for model_name in self.setup["models_list"].keys():
            k_model_id = TfModel.keyify_model_id(self.setup['models_list'][model_name]['model_id'])
            print(f"Testing model {model_name}, k_model id: {k_model_id}")
            if k_model_id not in self.known_models.keys():
                self.known_models[k_model_id]             = {}
                #self.known_models[model_name]["model_id"] = self.setup["models_list"][model_name]["model_id"]
                if self.debug:
                    print(f"Model {k_model_id}: is not yet known Testing model memory ressources requirements")
                    self.si.print_GPU_info()
                    
                if self.setup["models_list"][model_name]["model_type"] == "transformers":
                    ram_b4   = self.si.get_ram_free()
                    v_ram_b4 = self.si.get_v_ram_free()
                    self.instantiate_model(model_name)

                    mem_use_Gb     = float(   self.active_models[model_name].model.get_memory_footprint(return_buffers=True))/1024/1024/1024
                    model_ongpu    = False if self.active_models[model_name].model.device.type == "cpu" else True
                    delta_ram_gb   = max(0, ram_b4   - self.si.get_ram_free()  ) # min value is 0
                    delta_v_ram_gb = max(0, v_ram_b4 - self.si.get_v_ram_free()) # to avoid noise on the unused ram
                    self.known_models[k_model_id]["system_ram_gb"] = delta_ram_gb
                    self.known_models[k_model_id]["GPU_ram_gb"   ] = delta_v_ram_gb
                    if self.debug:
                        #print(f"Model {model_name} instantiated using: {delta_ram_gb:6.2} Gb of system RAM and: {delta_v_ram_gb:6.2} Gb of V RAM on the GPU, model on GPU:{model_ongpu} , memory use: {mem_use_Gb:6.2} Gb")
                        self.si.print_GPU_info()

                elif self.setup["models_list"][model_name]["model_type"] == "openAI":
                    self.known_models[k_model_id]["system_ram_gb"] = 0.0
                    self.known_models[k_model_id]["GPU_ram_gb"   ] = 0.0

                else:
                    print(f"ERROR: Unknown model type: {self.setup['models_list'][model_name]['model_type']}")

            with open(self.setup["known_models_file_name"], 'w') as f:  # store the known models dict to a file to avoid doing it again
                        json.dump(self.known_models, f, indent=4)
                        print(f"Model {k_model_id} added to known_models")

            self.deinstantiate_model(model_name)

        if self.debug:
            print(f"Models memory ressources requirements test complete")
            self.si.print_GPU_info()


    def instantiate_model(self, model_name:str) -> None:
        """instantiate the model defined in the set-up by adding it to the active model list and creating the corresponding embeddings"""
        model_id   = self.setup['models_list'][model_name]['model_id']
        
        if self.setup['models_list'][model_name]['model_type'] == "openAI":
            api_key                            = os.getenv('OPENAI_API_KEY')
            self.active_models[model_name]     = ChatOpenAI(openai_api_key=api_key, model_name=model_id)
            #self.active_embeddings[model_name] = OpenAIEmbeddings(openai_api_key=api_key)
            self.active_embeddings[model_name] = tiktoken.encoding_for_model(model_id)

        elif self.setup['models_list'][model_name]['model_type'] == "transformers":
            if self.debug:
                print(f"GPU state before model instantiation: {torch.cuda.is_available()}")
                self.si.print_GPU_info()

            self.active_models[model_name]     = TfModel(self.setup, self.si, model_name)
            self.active_embeddings[model_name] = TransformersEmbeddings(self.active_models[model_name].tokenizer)
            if self.debug:
                print(f"GPU state after model instantiation: {torch.cuda.is_available()}")
                self.si.print_GPU_info()
    

    def deinstantiate_model(self, model_name:str) -> None: 
        """deinstantiate the model provided in the argument end ensure proper deletion of all the model ressources"""
        print(f"Deinstantiating model {model_name}")
        if self.debug:
            print(f"GPU state before and after model deinstantiation: {torch.cuda.is_available()}")
            self.si.print_GPU_info()

        if model_name in self.active_embeddings:
            print(self.active_embeddings)
            self.active_embeddings[model_name].free()
            print(f"Embeddings {model_name} ref count: {sys.getrefcount(self.active_embeddings[model_name])}")
            del self.active_embeddings[model_name]
            #self.active_embeddings.pop(model_name)
            
        if model_name in self.active_models:
            print(self.active_models)
            self.active_models[    model_name].free()
            print(f"Model {model_name} ref count: {sys.getrefcount(self.active_models[model_name])}")
            del self.active_models[model_name]
            #self.active_models.pop(model_name)

        gc.collect()
        if torch.cuda.is_available():
            #print("-m--Emptying CUDA cache----")
            torch.cuda.empty_cache()

        if self.debug:
            self.si.print_GPU_info()
        
