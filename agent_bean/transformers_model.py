import sys
import gc
import torch
import transformers

from   typing                  import List, Dict
from   agent_bean.system_info  import SystemInfo


class TransformersEmbeddings:
    """This class wraps the HuggingFace transformers tokenizers to be uses like langchain's OpenAIEmbeddings"""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, local_files_only=False) -> None:
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> torch.Tensor:
        """Return the embeddings for the text"""
        return torch.tensor(self.tokenizer(text)['input_ids'])

    """def embed_documents(self, texts: List[str], chunk_size: Optional[int] = 0) -> List[List[float]]:
        \"""Return the embeddings for the query\"""
        #print(f"Embed DOC!")
        #print([self.tokenizer(text)['input_ids'] for text in texts])
        if chunk_size:
            print(f"ERROR: chunk_size: {chunk_size} NOT IMPLEMENTED YET")
            return [self.tokenizer(text)['input_ids'] for text in texts]
        else:
            return [self.tokenizer(text)['input_ids'] for text in texts]
    """
    #def embed_query(self, text: str) -> List[float]:
    def encode(self, text: str) -> List[int]:
        """Return the embeddings for the query"""
        #tok = self.tokenizer(text)
        #print(f"tok: {tok}")
        return self.tokenizer(text)['input_ids']
    
    def decode(self, tokens: List[int]) -> str:
        """Return the text for the tokens"""
        #print(f"tokens: {tokens}")
        return self.tokenizer.decode(tokens)
    
    def free(self) -> None:
        """Free the memory used by the embeddings"""
        self.tokenizer = None
    
       

class TfModel:
    """This class wraps the HuggingFace transformers pipeline class to allow to build a pipeline from the setting data"""
    def __init__(self, setup: dict, system_info:SystemInfo, model_name: str) -> None:
        self.setup                            = setup
        self.system_info                      = system_info
        self.model_name                       = model_name
        self.compute_dtype                    = torch.float16
        self.GPU_brand:str                    = self.system_info.get_gpu_brand()

        self.embeddings                       = None
        self.pipeline                         = None
        self.quant_type_4bit:bool             = None
        self.model_bits:int                   = None
        self.model                            = None
        self.tokenizer                        = None
        self.stopping_criteria                = None
        self.model_id:str                     = None
        self.k_model_id:str                   = None
        self.device                           = None

        self.do_sample:bool                   = True
        self.temperature:float                = 0.1     # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        self.top_p:float                      = 1
        self.top_k:float                      = 0
        self.frequency_penalty:float          = 0.6
        self.presence_penalty:float           = 0.0
        self.repetition_penalty:float         = 1.1     # without this output begins repeating
        self.stop:List[str]                   = ["\n"]
        self.max_new_tokens:int               = 512     # max number of tokens to generate in the output
        self.GPTQ_endings                     = ['GPTQ', 'gptq']
        self.GGUF_endings                     = ['GGUF', 'gguf']

        #print(f"GPU brand: {self.GPU_brand}")
        self.instantiate_pipeline()

    @staticmethod
    def keyify_model_id(model_id):
        """Clean the model_id into a string that can be used as a key"""
        return str(model_id).replace('/', '_-_')

    @staticmethod
    def de_keyify_model_id(cleaned_model_id):
        """Reverse the cleaning process to get the original model_id"""
        return cleaned_model_id.replace('_-_', '/')

    def instantiate_pipeline(self) -> None:
        """instantiate the pipeline defined in the set-up """
        self.device     = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        model_name      = self.model_name
                
        if self.setup['models_list'][model_name]['model_type'] == "transformers":
            # Prepare the parametters to instantiate the Transformers model
            self.model_id   = self.setup['models_list'][model_name]['model_id']
            self.k_model_id = self.keyify_model_id(self.model_id)

            for s in self.GPTQ_endings:
                if self.model_id.endswith(s):
                    #quantizer       = GPTQQuantizer(bits=4, dataset="c4", block_name_to_quantize = "model.decoder.layers", model_seqlen = 2048)
                    #quantized_model = quantizer.quantize_model(model, tokenizer)
                    raise ValueError("GPTQ models are not yet supported by Agent Bean")
            for s in self.GGUF_endings:
                if self.model_id.endswith(s):
                    raise ValueError("GGUF models are not yet supported by Agent Bean")

            # define the model kwargs
            pretrained_kwargs = {'device_map':'auto'}

            self.trust_remote_code = False
            if 'trust_remote_code' in self.setup['models_list'][model_name]:
                self.trust_remote_code = self.setup['models_list'][model_name]['trust_remote_code']
                pretrained_kwargs['trust_remote_code'] = self.trust_remote_code

            if 'flash_attn' in self.setup['models_list'][model_name]:
                self.flash_attn = self.setup['models_list'][model_name]['flash_attn']
                if self.flash_attn:
                    pretrained_kwargs['flash_attn'  ] = True
                    pretrained_kwargs['flash_rotary'] = True
                    pretrained_kwargs['fused_dense' ] = True

            if 'max_tokens' in self.setup['models_list'][model_name]:
                self.max_new_tokens = self.setup['models_list'][model_name]['max_tokens'] * 0.7

            if self.setup['transformers_local_only']:
                local_only                            = self.setup['transformers_local_only']
                pretrained_kwargs['local_files_only'] = local_only
            else:
                local_only                            = False

            if self.GPU_brand == 'NVIDIA':
                self.compute_dtype    = torch.bfloat16
            else:
                self.compute_dtype    = torch.float16
            
            pretrained_kwargs['torch_dtype'  ] = self.compute_dtype

            # check if the number of bits for quantization is set in setum model
            # set quantization configuration to load large model with less GPU memory
            # this requires the `bitsandbytes` library
            bnb_kwargs = {}

            if 'model_bits' in self.setup['models_list'][model_name]:
                self.model_bits = self.setup['models_list'][model_name]['model_bits']
                        
                if self.model_bits == 4:
                    # check if 4bit_quant_type in setum model
                    if '4bit_quant_type' in self.setup['models_list'][model_name]:
                        self.quant_type_4bit = self.setup['models_list'][model_name]['4bit_quant_type']
                        print(f"quant_type_4bit: {self.quant_type_4bit}")
                    else:
                        self.quant_type_4bit = 'nf4'

                    bnb_kwargs['disable_exllama'          ] = True
                    bnb_kwargs['get_loading_attributes'   ] = True
                    bnb_kwargs['load_in_4bit'             ] = True
                    bnb_kwargs['bnb_4bit_use_double_quant'] = True
                    bnb_kwargs['bnb_4bit_quant_type'      ] = self.quant_type_4bit
                    bnb_kwargs['bnb_4bit_compute_dtype'   ] = self.compute_dtype

                elif self.model_bits == 8:
                    bnb_kwargs['disable_exllama'          ] = True
                    bnb_kwargs['get_loading_attributes'   ] = True
                    bnb_kwargs['load_in_8bit'             ] = True
                    bnb_kwargs['bnb_8bit_use_double_quant'] = True
                    bnb_kwargs['bnb_8bit_compute_dtype'   ] = self.compute_dtype
                
            
            if bnb_kwargs:
                print(f" BnB kwargs: {bnb_kwargs}")
                bnb_config = transformers.BitsAndBytesConfig(**bnb_kwargs)
                pretrained_kwargs['quantization_config'] = bnb_config

            print(f" pretrained kwargs: {pretrained_kwargs}")
            self.model     = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, **pretrained_kwargs)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code   = self.trust_remote_code,
                local_files_only    = local_only,
            )

            self.embeddings        = TransformersEmbeddings(self.tokenizer)

            
            # set stopping criteria
            """
            stop_list         = ['\nHuman:', '\n```\n', '\nInstruction:', '\nUser:', '\n<|user|>' ]
            stop_token_ids    = [self.tokenizer(x)['input_ids'] for x in stop_list     ]
            stop_token_ids_LT = [torch.LongTensor(x) for x in stop_token_ids]  # convert to LongTensor for compatibility with model

            class StopOnTokens(transformers.StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    for stop_ids in stop_token_ids_LT:
                        if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                            return True
                    return False

            self.stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])
            """

            self.my_generation_config              = transformers.generation.GenerationConfig.from_model_config(self.model.config)
            self.my_generation_config.eos_token_id = self.model.config.eos_token_id

            #self.pipeline          = transformers.pipeline(
            #        model              = self.model, 
            #        tokenizer          = self.tokenizer,
            #        return_full_text   = True,                       # langchain expects the full text
            #        task               ='text-generation',
            #        stopping_criteria  = self.stopping_criteria,     # without this model rambles during chat
            #        do_sample          = True,
            #        temperature        = self.temperature,           # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            #        max_new_tokens     = self.max_new_tokens,        # max number of tokens to generate in the output
            #        repetition_penalty = self.repetition_penalty,    # without this output begins repeating
            #    )
        else:
            print(f"ERROR: NOT A TRANSFORMER MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
    def predict(self, prompt: str) -> List[str]:
        """predict the next token based on the prompt"""

        print(f"### PREDICT ### prompt length: {len(prompt)}, ### Gen CFG:\n{self.my_generation_config}\n")
        print(f"### PREDICT ### prompt: {prompt}")

        input_ids  = self.tokenizer.encode(prompt, return_tensors="pt"                           ).to(self.device)
        res_raw_t  = self.model.generate(  input_ids, generation_config=self.my_generation_config) 
        res_raw    = self.tokenizer.decode(res_raw_t[0], skip_special_tokens=True                )
        prompt_dec = self.tokenizer.decode(input_ids[0], skip_special_tokens=True                )
        
        print(f"\n### R E S  R A W ### Len: {len(res_raw)} ###: {res_raw}")
        res       = res_raw.replace(prompt_dec, '')

        if len(res) > 1:
            return res
        else:
            return ''
    

    def __del__(self):
        """Delete the transformers model."""
        print("---- Deleting Transformers model ----")
        if self is not None:
            if hasattr(self, 'embeddings'):   
                self.embeddings.free()
                del self.embeddings
            if hasattr(self, 'setup'):            del self.setup
            if hasattr(self, 'system_info'):      del self.system_info
            if hasattr(self, 'model_name'):       del self.model_name
            if hasattr(self, 'compute_dtype'):    del self.compute_dtype
            if hasattr(self, 'GPU_brand'):        del self.GPU_brand
            if hasattr(self, 'quant_type_4bit'):  del self.quant_type_4bit
            if hasattr(self, 'model_bits'):       del self.model_bits
            if hasattr(self, 'tokenizer'):        del self.tokenizer
            if hasattr(self, 'stopping_criteria'):del self.stopping_criteria
            if hasattr(self, 'model_id'):         del self.model_id
            if hasattr(self, 'stop'):             del self.stop
            if hasattr(self, 'GPTQ_endings'):     del self.GPTQ_endings
            if hasattr(self, 'GGUF_endings'):     del self.GGUF_endings
            if hasattr(self, 'model'):            
                print(f"TF Model __del__ refcounts: {sys.getrefcount(self.model)}")
                del self.model
            if hasattr(self, 'device'):           del self.device

        gc.collect()
        if torch.cuda is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            


    def free(self) -> None:
        """Free the memory used by the model"""
        print(f"Free the model {self.model_name}, using: {self.model.get_memory_footprint()/1024**3} Gb")
        #self.model.to_empty(device=self.device)
        self.embeddings.free()
        del self.embeddings
        del self.setup
        del self.system_info
        del self.model_name
        del self.compute_dtype
        del self.GPU_brand
 
        del self.pipeline
        del self.quant_type_4bit
        del self.model_bits
        del self.tokenizer
        del self.stopping_criteria
        del self.model_id
        del self.k_model_id
        del self.stop
        del self.GPTQ_endings
        del self.GGUF_endings
        print(f"TF Model free refcounts: {sys.getrefcount(self.model)}")
        del self.model
        del self.device

        gc.collect()


