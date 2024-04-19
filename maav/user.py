from maav.configuration import config
import maav.configuration as configuration
from maav.model import User_Model
from maav.tokenizer import User_Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class User:# not needed

    """
    This class offers an interface to use the model reccursively

    Methods:
        predict(): To predict on a single time step
        generate_response(): To generate text response reccursively given input query
    
    """
    def __init__(self) -> None:
        self.model = User_Model()
        self.tokenizer = User_Tokenizer()
    def predict(self, X_train):
        return self.model.predict(X_train)
    
    def generate_response(self, query="", length=1):

        query = configuration.split(query)
        query_sequence = self.tokenizer.text_to_sequences(query)
        query_sequence = pad_sequences([query_sequence], padding='pre', maxlen=config["MAX_SEQUENCE_LENGTH"], truncating='pre')[0]
        query_sequence = query_sequence.tolist()
        token_response = self.model.predict(query_sequence)
        #print(token_response)
        return config["TOKEN_TO_INSERT_BETWEEN"].join(self.tokenizer.sequence_to_text(token_response))
        return token_response
        


        for _ in range(length):
            token_prediction = self.model.predict(query_sequence)
            token_response.append(int(token_prediction))
            query_sequence.append(int(token_prediction))
            query_sequence = query_sequence[-config.MAX_SEQUENCE_LENGTH:]
        
        text_response = config.TOKEN_TO_INSERT_BETWEEN.join(self.tokenizer.sequence_to_text(token_response))
        return text_response



