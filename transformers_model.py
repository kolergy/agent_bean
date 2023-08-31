
import torch
import transformers

from   typing       import List, Optional
from   system_info  import SystemInfo


class TransformersEmbeddings:
    """This class wraps the HuggingFace transformers tokenizers to be uses like langchain's OpenAIEmbeddings"""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> torch.Tensor:
        """Return the embeddings for the text"""
        return torch.tensor(self.tokenizer(text)['input_ids'])

    def embed_documents(self, texts: List[str], chunk_size: Optional[int] = 0) -> List[List[float]]:
        """Return the embeddings for the query"""
        #print(f"Embed DOC!")
        #print([self.tokenizer(text)['input_ids'] for text in texts])
        if chunk_size:
            print(f"ERROR: chunk_size: {chunk_size} NOT IMPLEMENTED YET")
            return [self.tokenizer(text)['input_ids'] for text in texts]
        else:
            return [self.tokenizer(text)['input_ids'] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return the embeddings for the query"""
        tok = self.tokenizer(text)
        #print(f"tok: {tok}")
        return self.tokenizer(text)['input_ids']
    
    def __del__(self):
        """delete the tokenizer"""
        del self.tokenizer
        

class TfModel:
    """This class wraps the HuggingFace transformers pipeline class to allow to build a pipeline from the setting data"""
    def __init__(self, setup: dict, system_info:SystemInfo) -> None:
        self.setup             = setup
        self.system_info       = system_info

        self.compute_dtype     = torch.float16
        self.GPU_brand         = self.system_info.get_gpu_brand()

        self.pipeline          = None
        self.quant_type_4bit   = None
        self.model_bits        = None
        self.model             = None
        self.tokenizer         = None
        self.stopping_criteria = None
        #print(f"GPU brand: {self.GPU_brand}")
        self.instantiate_pipeline()

    def instantiate_pipeline(self) -> None:
        """instantiate the pipeline defined in the set-up """
        if self.setup['model']['model_type'] == "transformers":
            # Instantiate the Transformers model here
            # You will need to fill in the details based on how you want to use the Transformers library
            model_id = self.setup['model']['model_id']
            device   = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
            print(f"device: {device}, brand: {self.GPU_brand}")
            # check if the number of bits for quantization is set in setum model
            if 'model_bits' in self.setup['model']:
                self.model_bits = self.setup['model']['model_bits']
                        
            if self.GPU_brand == 'NVIDIA':
                self.compute_dtype    = torch.bfloat16
            else:
                self.compute_dtype    = torch.float16

            if self.model_bits == 4:
                # check if 4bit_quant_type in setum model
                if '4bit_quant_type' in self.setup['model']:
                    self.quant_type_4bit = self.setup['model']['4bit_quant_type']
                # set quantization configuration to load large model with less GPU memory
                # this requires the `bitsandbytes` library
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_4bit              = True,
                    bnb_4bit_quant_type       = self.quant_type_4bit,
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_compute_dtype    = self.compute_dtype,
                    disable_exllama           = True,
                    get_loading_attributes    = True,
                )
            
            elif self.model_bits == 8:
                # set quantization configuration to load large model with less GPU memory
                # this requires the `bitsandbytes` library
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_8bit              = True,
                    bnb_8bit_use_double_quant = True,
                    bnb_8bit_compute_dtype    = self.compute_dtype,
                    disable_exllama           = True,
                    get_loading_attributes    = True,
                )
            else:
                bnb_config = None


            if bnb_config:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_id,
                    #trust_remote_code   = True,
                    quantization_config = bnb_config,
                    torch_dtype         = self.compute_dtype,
                    device_map          = 'auto',
                )
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_id,
                    #trust_remote_code   = True,
                    torch_dtype         = self.compute_dtype,
                    device_map          = 'auto',
                )

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id,
            )

            self.embeddings = TransformersEmbeddings(self.tokenizer)

            # set stopping criteria
            stop_list         = ['\nHuman:', '\n```\n']
            stop_token_ids    = [self.tokenizer(x)['input_ids'] for x in stop_list     ]
            stop_token_ids_LT = [torch.LongTensor(x).to(device) for x in stop_token_ids]  # convert to LongTensor for compatibility with model

            class StopOnTokens(transformers.StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    for stop_ids in stop_token_ids_LT:
                        if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                            return True
                    return False

            self.stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

            self.pipeline          = transformers.pipeline(
                    model              = self.model, 
                    tokenizer          = self.tokenizer,
                    return_full_text   = True,                    # langchain expects the full text
                    task               ='text-generation',
                    stopping_criteria  = self.stopping_criteria,  # without this model rambles during chat
                    do_sample          = True,
                    temperature        = 0.1,                     # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                    max_new_tokens     = 512,                     # max number of tokens to generate in the output
                    repetition_penalty = 1.1,                      # without this output begins repeating
                )
            
    def predict(self, prompt: str,
                      max_tokens: int          = 1000,
                      temperature: float       =    0.01,
                      top_p: float             =    1,
                      top_k: float             =    0,
                      frequency_penalty :float =    0,
                      presence_penalty :float  =    0.6,
                      stop: str                = ["\n"]) -> str:
        """predict the next token based on the prompt"""
        print(f"### PREDICT ### prompt length: {len(prompt)}")
        if temperature <= 0.0: temperature = 0.01 # temp need to be strictly positive
        pre_prms  = {'return_tensors':"pt"                }
        fwd_prms  = {'max_new_tokens'  : max_tokens      ,
                     'temperature'     : temperature     ,
                     'top_p'           : top_p           ,
                     'top_k'           : top_k           ,
                     'do_sample'       : True            , }
        post_prms = {'clean_up_tokenization_spaces':True,  }
        res_raw   = self.pipeline.run_single(prompt, 
                                       preprocess_params  = pre_prms, 
                                       forward_params     = fwd_prms, 
                                       postprocess_params = post_prms)
        print(f"### R E S ###: {len(res_raw)}")
        res       = res_raw[0]['generated_text'].split('#~!|MODEL OUTPUT|!~#:')
        del res_raw
        del pre_prms
        del fwd_prms
        del post_prms
        del prompt

        if len(res) > 1:
            return [res[1]]
        else:
            return ['']
    
    def __del__(self):
        """delete the model"""
        del self.model
        del self.tokenizer
        del self.pipeline
        del self.embeddings
        del self.stopping_criteria
        del self.compute_dtype
        del self.GPU_brand
        del self.quant_type_4bit
        del self.model_bits
        del self.system_info
        if torch.cuda.is_available():
            print("-t--Emptying CUDA cache----")
            torch.cuda.empty_cache()


