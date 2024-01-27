class ChatInteraction:
    """
    This class represents a single chat interaction, including input, context,
    output, user rating, and number of tokens.
    """

    def __init__(self, input_text, context, output_text, user_rating=None, num_tokens=None):
        self.input_text = input_text
        self.context = context
        self.output_text = output_text
        self.user_rating = user_rating
        self.num_tokens = num_tokens

    def update_rating(self, rating):
        self.user_rating = rating

    def update_num_tokens(self, num_tokens):
        self.num_tokens = num_tokens


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

    def update_interaction_num_tokens(self, index, num_tokens):
        if index < len(self.interactions):
            self.interactions[index].update_num_tokens(num_tokens)

    def clear_interactions(self):
        self.interactions = []
