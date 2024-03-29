
import os
import sys

import gc
import json
import torch
import tiktoken

from   typing                            import List
from   langchain_community.chat_models   import ChatOpenAI
from   agent_bean.system_info            import SystemInfo
from   agent_bean.transformers_model     import TfModel     , TransformersEmbeddings
from   agent_bean.ollama_model           import OllamaModel , OllamaEmbeddings
from   agent_bean.mistral_model          import MistralModel, MistralEmbeddings
from   transformers                      import GenerationConfig
#from agent_bean.google_vertexai_model import VertexAIModel, VertexAIEmbeddings

class ModelsManager():
    """This class is used to manage the models and their resource usage.
    It handles instantiation, parameter setting, prediction, and resource management
    for various machine learning models, ensuring that the system's memory constraints
    are respected.
    """
    def __init__(self, setup: dict, si: SystemInfo) -> None:

        self.setup_update(setup)
        self.setup                    = setup
        self.si                       = si
        self.debug                    = setup['debug']
        self.active_models            = {}
        self.active_embeddings        = {}
        self.known_models             = {}
        self.openai_params_list       = ["temperature", "max_tokens"]
        self.mistral_params_list      = ["temperature", "max_tokens", "top_p", "stream", "safe_prompt", "random_seed",]
        self.transformers_params_list = ["temperature", "max_tokens", "stop", "presence_penalty", "frequency_penalty", "top_p", "top_k", "repetition_penalty", 
                                         "do_sample", "max_new_tokens", "min_length", "min_new_tokens", "early_stopping", "max_time", "num_beams", "num_beam_groups",
                                         "penalty_alpha", "use_cache", "typical_p", "epsilon_cutoff", "eta_cutoff", "diversity_penalty", "repetition_penalty", 
                                         "encoder_repetition_penalty", "length_penalty", "no_repeat_ngram_size", "bad_words_ids", "force_words_ids", "renormalize_logits",
                                         "constraints", "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values", "exponential_decay_length_penalty",
                                         "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids", "sequence_bias", "guidance_scale", "low_memory", "num_return_sequences",
                                         "output_attentions", "output_hidden_states", "output_scores", "return_dict_in_generate", "pad_token_id", "bos_token_id", "eos_token_id",
                                         "encoder_no_repeat_ngram_size", "decoder_start_token_id", "num_assistant_tokens", "num_assistant_tokens_schedule", "generation_kwargs",
                                         "_from_model_config", "transformers_version"]

        #self.test_models_resources_reqs()
    

    def setup_update(self, setup: dict) -> None:
        """Update the setup configuration for the model manager.

        This method updates the internal setup dictionary and debug flag. It also
        handles the reloading of models by deinstantiating any active models if
        the 'reload_models' flag is set in the setup.

        Args:
            setup (dict): A dictionary containing the new setup configuration.
        """
        self.setup = setup
        self.debug = setup['debug']
        print(f"ModelsManager.setup_update: models={self.setup['models_list'].keys()}")
        if os.path.exists(self.setup["known_models_file_name"]):
            with open(self.setup["known_models_file_name"]) as f:
                self.known_models  = json.load(f)
        if 'reload_models' in self.setup.keys():
            if self.setup['reload_models']:
                if len(self.active_models) > 0:
                    for m in self.active_models.keys():
                        self.deinstantiate_model(m)

    def get_available_models(self) -> List[str]:
        """Return a list of available model names."""
        return list(self.setup['models_list'].keys())
    
    def model_need(self, model_name:str) -> bool:
        """Ensure a model is instantiated and ready for use.

        If the model is not already instantiated, this method checks memory resources
        and instantiates the model if needed. It may also remove other instantiated
        models to free up resources.

        Args:
            model_name (str): The name of the model to instantiate.

        Returns:
            bool: True if the model is successfully instantiated or already active,
                  False otherwise.

        Raises:
            RuntimeError: If there are not enough resources to instantiate the model.
        """
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
            #if self.debug:
            #    print(f"Model {model_name} already instantiated")
            return True
        

    def set_model_params(self, model_name:str, params:dict) -> None:
        """Set parameters for a specified model.

        This method updates the model's parameters if the model is active and the
        parameters are applicable to the model type.

        Args:
            model_name (str): The name of the model for which to set parameters.
            params (dict): A dictionary of parameters to set for the model.
        """
        if self.model_need(model_name):
            print(f" model Parametrisation: model type: {type(self.active_models[model_name])}")
            if isinstance(self.active_models[model_name],ChatOpenAI):
                for p in params.keys():
                    if p in self.openai_params_list:
                        print(f"setting param: {p}, value: {params[p]}")
                        setattr(self.active_models[model_name], p, params[p])
                    else:
                        print(f"WARNING: Ignoring unknown param: {p} for model type: {type(self.active_models[model_name])}")

            elif isinstance(self.active_models[model_name],MistralModel):
                for p in params.keys():
                    if p in self.mistral_params_list:
                        print(f"setting param: {p}, value: {params[p]}")
                        setattr(self.active_models[model_name], p, params[p])
                    else:
                        print(f"WARNING: Ignoring unknown param: {p} for model type: {type(self.active_models[model_name])}")

            elif isinstance(self.active_models[model_name], TfModel):
                cur_params = self.active_models[model_name].my_generation_config.to_dict()
                cur_params.update(params)
                for p in list(cur_params.keys()):
                    if p not in self.transformers_params_list:
                        print(f"WARNING: Ignoring unknown param: {p} for model type: {type(self.active_models[model_name])}")
                        del cur_params[p]
                    elif p == "temperature":   # if temperature is set to > 0 then it necessitates sampling
                            if cur_params['temperature'] > 0:
                                cur_params['do_sample'] = True
                            else:
                                cur_params['do_sample'] = False

                cur_params['pad_token_id'] = self.active_models[model_name].tokenizer.eos_token_id   # to avoid the warning: "Setting `pad_token_id` to `eos_token_id`
                self.active_models[model_name].my_generation_config = GenerationConfig(**cur_params)
            else:
                print(f"ERROR: Unknown model type: {type(self.active_models[model_name])}")
        else:
            print(f"ERROR: can not set params Model {model_name} could not be instantiated")


    def predict(self, model_name:str, prompt:str ) -> str:
        """Generate a prediction using a specified model.

        This method uses the active model to generate a prediction based on the
        provided prompt.

        Args:
            model_name (str): The name of the model to use for prediction.
            prompt (str): The input prompt for the model.

        Returns:
            str: The prediction result from the model.
        """
        if self.model_need(model_name):
            res =  self.active_models[model_name].predict(prompt)
            print(f"predict result: {res}") 
            return [res]
        else:
            return None


    def decode(self, model_name:str, tokens:List[float]) -> str:
        """Decode a sequence of tokens using a specified model's embeddings.

        Args:
            model_name (str): The name of the model to use for decoding.
            tokens ([float]): A list of tokens to decode.

        Returns:
            str: The decoded text from the tokens.
        """
        if self.model_need(model_name):
            return self.active_embeddings[model_name].decode(tokens)
        else:
            return None


    def get_embeddings(self, model_name:str, text:str) -> torch.tensor:
        """Retrieve embeddings for a given text using a specified model.

        Args:
            model_name (str): The name of the model to use for generating embeddings.
            text (str): The input text to encode into embeddings.

        Returns:
            torch.tensor: The tensor containing the embeddings for the input text.
        """
        if self.model_need(model_name):
            return self.active_embeddings[model_name].encode(text)
            #return self.active_embeddings[model_name].embed_query(text)
        else:
            return None

    def get_encoder(self, model_name:str):
        """Retrieve the encoder for a specified model.

        Args:
            model_name (str): The name of the model for which to retrieve the encoder.

        Returns:
            object: The encoder object for the specified model.
        """
        if self.model_need(model_name):
            return self.active_embeddings[model_name]
        else:
            return None

    def manage_mem_resources(self, model_name:str) -> bool:
        """Check and manage memory resources for a model instantiation.

        This method compares the memory requirements of a model against available
        system and GPU memory. It may de-instantiate other models to free up
        resources if necessary.

        Args:
            model_name (str): The name of the model for which to manage resources.

        Returns:
            bool: True if there are enough resources to instantiate the model,
                  False otherwise.
        """
        k_model_id = TfModel.keyify_model_id(self.setup['models_list'][model_name]['model_id'])
        #if self.debug:
        #    print(f"Checking memory resources for model {k_model_id}")
        #    self.si.print_GPU_info()
        if k_model_id not in self.known_models:
            print(f"WARNING: Model {k_model_id} not in known_models list, running blind de instanciating everything just in case") 
            keys = list(self.active_models.keys())  # avoid -> RuntimeError: dictionary changed size during iteration
            for m in keys:
                print(f"####### {type(self.active_models[m])}")
                if type(self.active_models[m]) == "TFModel":
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
        """Free up system and GPU memory to meet required thresholds.

        This method attempts to de-instantiate models to free up the specified
        amount of system and GPU memory.

        Args:
            required_free_ram_gb (float): The required amount of free system RAM in GB.
            required_free_v_ram_gb (float): The required amount of free GPU VRAM in GB.

        Returns:
            bool: True if the required resources are successfully freed, False otherwise.
        """
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

    def instantiate_model(self, model_name:str) -> None:
        """instantiate the model defined in the set-up by adding it to the active model list and creating the corresponding embeddings"""
        model_id   = self.setup['models_list'][model_name]['model_id']
        
        if self.setup['models_list'][model_name]['model_type'] == "openAI":
            OAI_api_key                        = os.getenv('OPENAI_API_KEY')
            self.active_models[model_name]     = ChatOpenAI(openai_api_key=OAI_api_key, model_name=model_id)
            self.active_embeddings[model_name] = tiktoken.encoding_for_model(model_id)
        elif self.setup['models_list'][model_name]['model_type'] == "ollama":
            self.active_models[model_name]     = OllamaModel(self.setup, self.si, model_name=model_id)
            self.active_embeddings[model_name] = OllamaEmbeddings(self.active_models[model_name].tokenizer)

        if self.setup['models_list'][model_name]['model_type'] == "Mistral_API":
            Mistral_api_key                    = os.getenv('MISTRAL_API_KEY')
            self.active_models[model_name]     = MistralModel(self.setup, self.si,Mistral_api_key=Mistral_api_key, model_name=model_id)
            self.active_embeddings[model_name] = MistralEmbeddings(Mistral_api_key=Mistral_api_key, model_name= "mistral-embed")
        #elif self.setup['models_list'][model_name]['model_type'] == "vertexai_api":
        #    VertexAI_project_id                = os.getenv('VERTEXAI_PROJECT_ID')
        #    VertexAI_location                  = os.getenv('VERTEXAI_LOCATION')
        #    self.active_models[model_name]     = VertexAIModel(project_id=VertexAI_project_id, location=VertexAI_location, model_id=model_id)
        #    self.active_embeddings[model_name] = VertexAIEmbeddings(project_id=VertexAI_project_id, location=VertexAI_location, model_id=model_id)#

        elif self.setup['models_list'][model_name]['model_type'] == "transformers":
            self.active_models[model_name]     = TfModel(self.setup, self.si, model_name)
            self.active_embeddings[model_name] = TransformersEmbeddings(self.active_models[model_name].tokenizer)
        #elif self.setup['models_list'][model_name]['model_type'] == "vertexai_api":
        #    VertexAI_project_id                = os.getenv('VERTEXAI_PROJECT_ID')
        #    VertexAI_location                  = os.getenv('VERTEXAI_LOCATION')
        #    self.active_models[model_name]     = VertexAIModel(project_id=VertexAI_project_id, location=VertexAI_location, model_id=model_id)
        #    self.active_embeddings[model_name] = VertexAIEmbeddings(project_id=VertexAI_project_id, location=VertexAI_location, model_id=model_id)#

    

    def deinstantiate_model(self, model_name:str) -> None: 
        """deinstantiate the model provided in the argument end ensure proper deletion of all the model ressources"""
        print(f"Deinstantiating model {model_name}")
        #if self.debug:
        #    print(f"GPU state before and after model deinstantiation: {torch.cuda.is_available()}")
        #    self.si.print_GPU_info()

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

        #if self.debug:
        #    self.si.print_GPU_info()
        
