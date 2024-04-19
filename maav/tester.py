# not needed

from maav.configuration import config
import maav.configuration as configuration
from maav.model import User_Model
from maav.tokenizer import User_Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import csv
from maav.configuration import config
import maav.configuration as configuration
from maav.dataset_preprocessor import Trainer_Dataset
from maav.tokenizer import Traininer_Tokenizer
from maav.model import Tester_Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

class Tester:

    """
    This class offers an interface to use the model reccursively

    Methods:
        predict(): To predict on a single time step
        generate_response(): To generate text response reccursively given input query
    
    """
    def __init__(self, dataset) -> None:
        self.model = Tester_Model()
        self.dataset = dataset
        self.dataset.y_train_index = 50
        #self.dataset.load()
        #self.tokenizer = User_Tokenizer()
    
    def generate_predictions(self):
        predictions = self.model.predict(pad_sequences(self.dataset.X_train, padding='pre', maxlen=config["MAX_SEQUENCE_LENGTH"], truncating='pre'))
        for _ in predictions:
            if _[0] > 0.5:
                _[0] = 1
            else:
                _[0] = 0
        self.predictions = predictions
    
    def save_as_csv(self):
        with open(config["TEST_RESULT_OUTPUT_FILE_PATH"], mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['PassengerId', 'Survived'])
            print(len(self.dataset.ids), len(self.predictions))
            for i, prediction in enumerate(self.predictions):
                writer.writerow([self.dataset.ids[i], prediction[0]])

