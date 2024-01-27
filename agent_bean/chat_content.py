import uuid

class TextContent:

    def __init__(self, encoder, text:str, tokenised_text=None):
        self.text           = text
        self.tokenised_text = tokenised_text
        self.num_tokens     = len(tokenised_text) if tokenised_text is not None else None
        self.num_words      = len(text.split())
        self.encoder        = encoder
        self.uuid           = uuid.uuid4()
        if self.tokenised_text is None:
            self.encode()

    def encode(self) -> None:
        """
        Encode the text using the model's tokenizer.
        """
        self.tokenised_text = self.encoder(self.text)
        self.num_tokens     = len(self.tokenised_text)

class ChatInteraction:
    """
    This class represents a single chat interaction, including input, context,
    output, user rating, and number of tokens.
    """

    def __init__(self, encoder:str, input_text:str, context:[str], output_text:[str]=[],user_rating:int=None):
        self.encoder          =  encoder
        self.encoder          =  None
        self.input_text       =  TextContent(encoder, input_text)
        self.context          = [TextContent(encoder, c) for c in context]
        self.output_text      =  TextContent(encoder, output_text)
        self.user_rating      =  user_rating
        

    def update_rating(self, rating):
        self.user_rating = rating



class ChatContent:
    """
    This class is designed to store and manipulate the inputs and outputs of a language model (LLM),
    with each chat interaction stored as a separate ChatInteraction object.
    """

    def __init__(self):
        self.interactions = []

    def add_interaction(self, input_text, context, output_text, user_rating=None, num_tokens=None):
        interaction = ChatInteraction(input_text, context, output_text, user_rating, num_tokens)
        self.interactions.append(interaction)

    def get_interactions(self):
        return self.interactions

    def get_interaction(self, index):
        if index < len(self.interactions):
            return self.interactions[index]
        else:
            return None

    def update_interaction_rating(self, index, rating):
        if index < len(self.interactions):
            self.interactions[index].update_rating(rating)

    def update_interaction_num_tokens(self, index, text: str, tokenizer):
        """
        Update the number of tokens for a specific interaction based on the text and tokenizer.

        :param index: The index of the interaction to update.
        :param text: The text to tokenize.
        :param tokenizer: The tokenizer instance from the model.
        """
        if index < len(self.interactions):
            self.interactions[index].update_num_tokens(text, tokenizer)

    def clear_interactions(self):
        self.interactions = []
