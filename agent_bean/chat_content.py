import uuid

from   agent_bean.models_manager         import ModelsManager


class TextContent:

    def __init__(self, encoder, text:str, tokenised_text=None):
        self.text           = text
        print(f"TextContent: {text}")
        self.num_words      = len(text.split())
        self.uuid           = uuid.uuid4()
        if tokenised_text is None:
            self.tokenised_text = encoder.encode(self.text)
            self.num_tokens     = len(self.tokenised_text)
        else:
            self.tokenised_text = tokenised_text
            self.num_tokens     = len(tokenised_text)   

        
class ChatInteraction:
    """
    This class represents a single chat interaction, including input, context,
    output, user rating, and the model used to generate the output.
    """

    def __init__(self, encoder, action_name:str, model_name:str, input_text:str, context:[str], output_text:[str]=[],user_rating:int=None):
        if type(input_text) is str:
            input_text = [input_text]
        self.input_text       = [TextContent(encoder, c) for c in input_text ]
        self.context          = [TextContent(encoder, c) for c in context    ]
        self.output_text      = [TextContent(encoder, c) for c in output_text]
        self.user_rating      =  user_rating
        self.model_name       =  model_name
        self.action_name      =  action_name
        
        
    def update_rating(self, rating):
        self.user_rating = rating



class ChatContent:
    """
    This class is designed to store and manipulate the inputs and outputs of a language model (LLM),
    with each chat interaction stored as a separate ChatInteraction object.
    """

    def __init__(self, models_manager:ModelsManager):
        self.models_manager = models_manager
        self.interactions   = []

    def add_interaction(self, action_name:str, model_name:str, input_text:str, context:[str], output_text:[str]=[],user_rating:int=None):
        self.interactions.append(ChatInteraction(
            encoder     = self.models_manager.get_encoder(model_name), 
            action_name = action_name, 
            model_name  = model_name, 
            input_text  = input_text, 
            context     = context, 
            output_text = output_text, 
            user_rating = user_rating))

    def update_interaction_rating(self, index, rating):
        if index < len(self.interactions):
            self.interactions[index].update_rating(rating)

    def get_context(self) -> [str]:
        context = self.interactions[0].context if len(self.interactions) > 0 else []
        print(f"get context 0: {context}")
        #context.append(f(i) for i in self.interactions for f in (f.input_text.text, f.output_text))
        for i in self.interactions:
            print(f" i.input_text: {i.input_text}; i.output_text: {i.output_text}")
            print(f" i.input_text[0].text: {i.input_text[0].text}; i.output_text[0].text: {i.output_text[0].text}")
            #context.append(j.text for j in i.input_text )
            #context.append(j.text for j in i.output_text)
            for j in i.input_text:
                context.append(j.text)
            for j in i.output_text:
                context.append(j.text)

        print(f"get context: {context}")
        return context
    


    