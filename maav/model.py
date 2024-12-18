import numpy as np
import pickle
import tensorflow as tf
from maav.configuration import config
import maav.configuration as configuration
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, SimpleRNN, Masking
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import Callback
from maav.dataset_preprocessor import Dataset
from maav.tokenizer import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import maav.ran_optimizer as op


class Model:
    """
    The class which defines the model architecture and configurations

    Methods:
        load(): To load the saved model
    """
    def __init__(self) -> None:

        # Define the model
        class GatingLayer(tf.keras.layers.Layer):# not needed
            def __init__(self, units):
                super(GatingLayer, self).__init__()
                self.units = units
                self.gate = tf.keras.layers.Dense(units, activation='sigmoid')

            def call(self, inputs):
                gate_values = self.gate(inputs)
                gated_inputs = gate_values * inputs
                return gated_inputs

            
        self.model = Sequential([
            Dense(config["MAX_SEQUENCE_LENGTH"], activation='relu'),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(config["MAX_OUTPUT_SEQUENCE_LENGTH"], activation='relu')
        ])

        optimizer = optimizers.Adam(learning_rate=config["LEARNING_RATE"])

        # Compile the mode
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    
    def load(self):
        self.model = load_model("saved_model/sequential")

    
class Trainer_Model(Model):
    """
    This class provides an easy interface to Model class

    Specifically designed to training

    Methods:
        train_model(): To train the model
        save_model(): To save the model

    """
    def __init__(self) -> None:
        super().__init__()
        self.history = None
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    def train_model_using_adam(self, X_train, y_train, epochs=10, batch_size=32, verbose=1):
        if verbose == 0:
            for _ in range(epochs):
                self.history = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, batch_size=int(batch_size), verbose=verbose)
                configuration.progress_bar("Traininng", _, epochs)
        self.history = self.model.fit(np.array(X_train), np.array(y_train), epochs=int(epochs), batch_size=int(batch_size), verbose=verbose)
        return self.history
    
    def train_model(self, X_train, y_train, epochs=10, batch_size=32, verbose=1):
        print(epochs)
        optm = op.Ran_Optimizer(self.model, X_train, y_train, int(epochs))
        print("Hello\n\n\n")
        self.history = optm.train(int(epochs))
        print("Hai")
        print("Hello\n\n\n")
        return

    def save_model(self):
        self.model.save("saved_model/sequential")
        configuration.os.makedirs("saved_model/training_log", exist_ok=True)
        with open("saved_model/training_log/loss.history", 'wb') as file:
            pickle.dump(self.history.history['loss'], file)
        configuration.os.makedirs("saved_model/training_log", exist_ok=True)
        with open("saved_model/training_log/val_loss.history", 'wb') as file:
            try:
                pickle.dump(self.history.history['val_loss'], file)
            except KeyError:
                return

class User_Model(Model):# not needed
    """
    This class provides an easy interface to Model class

    Specifically designed to using the model

    """
    def __init__(self) -> None:
        super().__init__()
        self.load()
        print(f"\n\nLoaded model weight: {self.model.layers[1].get_weights()[0][0][0]}")
    def predict(self, X_new):
        predictions = self.model.predict(np.array([X_new]))[0]
        splitted = []
        index=0
        for _ in range((len(predictions) - config["VOCAB_NEURONS"])+1):
            splitted.append(predictions[index : index+config["VOCAB_NEURONS"]])
            index = index + config["VOCAB_NEURONS"]
        decimalsd = []
        
        # Convert the binary string to an integer

        #return decimalsd
        for each in splitted:
            binary_string = ''.join(map(str, [1 if num > 0.5 else 0 for num in each]))
            try:
                result = int(binary_string, 2)
            except ValueError:
                break

            decimalsd.append(result)
        return decimalsd

class Advanced_Trainer_Model(Trainer_Model):# not needed
    """
    Class with more advanced features than normal Trainer_Model class
    
    """
    def __init__(self) -> None:
        super().__init__()
        
    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(np.array(X_train), np.array(y_train), epochs=int(epochs), validation_split=config["VALIDATION_SPLIT"], batch_size=int(batch_size))
        

class Tester_Model(User_Model):# not needed
    def __init__(self) -> None:
        super().__init__()
    def predict(self, X_test):
        return self.model.predict(X_test)
    def generate_predictions(self, X_test):
        predictions = self.model.predict(np.array(X_test))
        print(predictions)