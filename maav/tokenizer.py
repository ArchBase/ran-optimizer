# not needed

from maav.configuration import config
import maav.configuration as configuration
import pickle
import os


class Tokenizer:
    """
    This is the global tokenizer for this model

    Methods:
        fit_on_chars(): Method to fit model to a list of string
        text_to_sequences(): To convert given string to tokenized sequence
        sequence_to_text(): To convert given sequence of tokens to string list
        save(): To save the tokenizer learned values
        load(): To load the tokenizer learned values

    """
    def __init__(self) -> None:
        self.string_to_token = {}
        self.token_to_string = {}
        self.last_index = 0
        self.token_float = config["TOKEN_INDICE_INIT_FACTOR"]
        self.commonality = 0
        
    def fit_on_chars(self, char_sequence=[]):
        for char in char_sequence:
            if char not in self.string_to_token:
                self.last_index += 1
                self.token_float += config["TOKEN_INDICE_UP_FACTOR"]
                self.string_to_token[char] = configuration.to_key(self.token_float)
                config["LAST_TOKEN_INDEX"] = self.token_float
                self.token_to_string[configuration.to_key(self.token_float)] = char
            else:
                self.commonality += 1
        config["VOCABULARY_SIZE"] = self.last_index + 1
        print(f"\t\tVocabulary size: {config['VOCABULARY_SIZE']}")
        print(f"\t\tCommonality: {self.commonality}\n")
        
    def text_to_sequences(self, text=[]):
        sequences = []
        char_sequence = text

        for each_char in char_sequence:
            try:
                sequences.append(configuration.to_value(self.string_to_token[each_char]))
            except KeyError:
                sequences.append(0)
        return sequences

    def sequence_to_text(self, sequences=[]):
        text = []
        for i in sequences:
            i = int(i)
            i = round(i/config["TOKEN_INDICE_UP_FACTOR"])*config["TOKEN_INDICE_UP_FACTOR"]
            try:
                text.append(self.token_to_string[configuration.to_key(i)])
            except KeyError as e:
                #text.append("<Unknown" + str(i) + ">")
                continue
        for index, word in enumerate(text):
            if word == "<Unknown0>":
                #print("deleted")
                del text[index]
                
        return text
    def save(self):
        os.makedirs("saved_model/tokenizer", exist_ok=True)
        with open("saved_model/tokenizer/pair_index.dict", 'wb') as file:
            pickle.dump(self.string_to_token, file)
        with open("saved_model/tokenizer/index_pair.dict", 'wb') as file:
            pickle.dump(self.token_to_string, file)
        with open("saved_model/tokenizer/last_index.int", 'wb') as file:
            pickle.dump(self.last_index, file)
    def load(self):
        print("\nLoading tokenizer.")
        with open("saved_model/tokenizer/pair_index.dict", 'rb') as file:
            self.string_to_token = pickle.load(file)
        with open("saved_model/tokenizer/index_pair.dict", 'rb') as file:
            self.token_to_string = pickle.load(file)
        with open("saved_model/tokenizer/last_index.int", 'rb') as file:
            self.last_index = pickle.load(file)
        
        
    

class Traininer_Tokenizer(Tokenizer):
    """
    class to provide an easy interface to deal with Tokenizer

    Specifically designed for training

    """
    def __init__(self) -> None:
        super().__init__()
    
class User_Tokenizer(Tokenizer):
    """
    class to provide an easy interface to deal with Tokenizer

    Specifically designed for using the model

    """
    def __init__(self) -> None:
        super().__init__()
        self.load()


