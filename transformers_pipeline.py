
import torch
import transformers

from   torch        import cuda, bfloat16, float16
from   system_info  import SystemInfo
from   transformers import StoppingCriteria, StoppingCriteriaList


class TfPipeline:
    """This class wraps the HuggingFace transformers pipeline class to allow to build a pipeline from the setting data"""
    def __init__(self, setup: dict, system_info:SystemInfo) -> None:
        self.setup             = setup
        self.system_info       = system_info

        self.compute_dtype     = float16
        self.GPU_brand         = self.system_info.get_gpu_brand()

        self.pipeline          = None
        self.quant_type_4bit   = None
        self.model_bits        = None
        self.model             = None
        self.tokenizer         = None
        self.stopping_criteria = None

        self.instantiate_pipeline()

    def instantiate_pipeline(self) -> None:
        """instantiate the pipeline defined in the set-up """
        if self.setup['model']['model_type'] == "transformers":
            # Instantiate the Transformers model here
            # You will need to fill in the details based on how you want to use the Transformers library
            model_id = self.setup['model']['model_id']
            device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
            
            # check if the number of bits for quantization is set in setum model
            if 'model_bits' in self.setup['model']:
                self.model_bits = self.setup['model']['model_bits']
                        
            if self.GPU_brand == 'NVIDIA':
                self.compute_dtype    = bfloat16
            else:
                self.compute_dtype    = float16

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
                    bnb_4bit_compute_dtype    = self.compute_dtype
                )
            
            elif self.model_bits == 8:
                # set quantization configuration to load large model with less GPU memory
                # this requires the `bitsandbytes` library
                bnb_config = transformers.BitsAndBytesConfig(
                    load_in_8bit              = True,
                    bnb_8bit_use_double_quant = True,
                    bnb_8bit_compute_dtype    = self.compute_dtype
                )
            else:
                bnb_config = None

            model_config = transformers.AutoConfig.from_pretrained(
                model_id,
            )

            if bnb_config:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code   = True,
                    config              = model_config,
                    quantization_config = bnb_config,
                    device_map          = 'auto',
                )
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code   = True,
                    config              = model_config,
                    device_map          = 'auto',
                )

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id,
            )


            # set stopping criteria
            stop_list         = ['\nHuman:', '\n```\n']
            stop_token_ids    = [self.tokenizer(x)['input_ids'] for x in stop_list     ]
            stop_token_ids_LT = [torch.LongTensor(x).to(device) for x in stop_token_ids]  # convert to LongTensor for compatibility with model

            class StopOnTokens(StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    for stop_ids in stop_token_ids_LT:
                        if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                            return True
                    return False

            self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])

            self.pipeline          = transformers.pipeline(
                    model              = self.model, 
                    tokenizer          = self.tokenizer,
                    return_full_text   = True,                    # langchain expects the full text
                    task               ='text-generation',
                    stopping_criteria  = self.stopping_criteria,  # without this model rambles during chat
                    temperature        = 0.1,                     # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                    max_new_tokens     = 512,                     # max number of tokens to generate in the output
                    repetition_penalty = 1.1                      # without this output begins repeating
                )