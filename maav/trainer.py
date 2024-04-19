from maav.configuration import config
import maav.configuration as configuration
from maav.dataset_preprocessor import Trainer_Dataset
from maav.tokenizer import Traininer_Tokenizer
from maav.model import Trainer_Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time


class Trainer:
    """
    This is the class used to train the model, this class provides access to train
    new models as well as to train or fine tune existing saved model

    Args:
        format: This is a string specifying wheather to train a new model or to train existing one
                format="new" means training new model
                format="*" means fine tunning exissting model

    methods:
        train_model(): The function used to train the model
    """
    def __init__(self, format=""):


        if format == "new":# Train new model
            self.dataset = Trainer_Dataset()
            self.model = Trainer_Model()
        else:# Train existing model
            self.dataset = Trainer_Dataset()
            self.model = Trainer_Model()

            # Load existing model
            self.model.load()
            print(f"\n\nLoaded model weight: {self.model.model.layers[1].get_weights()[0][0][0]}")
    def train_model_using_adam(self, epochs=10, batch_size=32, save=False, process_dataset=False):
        #Load dataset and Tokenizers
        self.dataset.load()
        #print(self.dataset.get_processed_dataset())
        #return
        # The training section
        print("\nStarting training.")
        try:
            start_time = time.time()
            self.model.train_model(tf.cast(tf.constant(self.dataset.X_train), tf.int32), tf.cast(tf.constant(self.dataset.y_train), tf.int32), epochs=epochs, batch_size=batch_size)
            end_time = time.time()
            time_taken = end_time - start_time
            print("Time taken by the optimization algorithm: {:.6f} seconds".format(time_taken))
        except KeyboardInterrupt:
            print("\nYou forcefully stopped the training process thus training logs are not available\n")
            option = input("\nDo you want to save the current learned weights?(y/n) ")
            if option != 'y':
                return

        # Save the model if save=True
        if save:
            self.model.save_model()
            print(f"\n\nSaved model weight: {self.model.model.layers[1].get_weights()[0][0][0]}\nModel saving complete")

    def train_model(self, epochs=10, batch_size=32, save=False, process_dataset=False):

        #Load dataset and Tokenizers
        self.dataset.load()
        #print(self.dataset.get_processed_dataset())
        #return
        # The training section
        print("\nStarting training.")
        try:
            start_time = time.time()
            self.model.train_model(tf.cast(tf.constant(self.dataset.X_train), tf.int32), tf.cast(tf.constant(self.dataset.y_train), tf.int32), epochs=epochs, batch_size=batch_size)
            end_time = time.time()
            time_taken = end_time - start_time
            print("Time taken by the optimization algorithm: {:.6f} seconds".format(time_taken))
        except KeyboardInterrupt:
            print("\nYou forcefully stopped the training process thus training logs are not available\n")
            option = input("\nDo you want to save the current learned weights?(y/n) ")
            if option != 'y':
                return

        # Save the model if save=True
        if save:
            self.model.save_model()
            print(f"\n\nSaved model weight: {self.model.model.layers[1].get_weights()[0][0][0]}\nModel saving complete")

        
    





class  Augmented_Trainer(Trainer):# not needed
    def __init__(self, try_out_times=0, batch_size=0, epoch_per_aug=0):
        super().__init__(format="new")
        self.try_out_times = try_out_times
        self.batch_size = batch_size
        self.epoch_per_aug = epoch_per_aug
        self.logs = []
        
    def start_training(self):
        self.dataset.load()
        try:
            for _ in range(self.try_out_times):
                print(f"\nAug: {_+1}/{self.try_out_times}")
                history = self.model.train_model(tf.cast(tf.constant(self.dataset.X_train), tf.int32), tf.cast(tf.constant(self.dataset.y_train), tf.int8), epochs=self.epoch_per_aug, batch_size=self.batch_size, verbose=1)
                print(f"last val_loss: {history.history['val_loss'][-1]}")
                self.logs.append({'val_loss': history.history['val_loss'][-1], 'X_train': self.dataset.X_train, 'y_train': self.dataset.y_train})#, 'model': self.model})
                self.dataset.shuffle()
                self.clear_weights()
        except KeyboardInterrupt:
            print("You forcefully stopped data augmented training. Finding best out of current augmentations")

        print("Calculating best performance..")
        min = self.logs[0]
        #print(self.logs)
        for _ in self.logs:
            if _['val_loss'] < min['val_loss']:
                min = _
        print(f"Best dataset split found with val_loss of {min['val_loss']}")
        self.dataset.X_train = min['X_train']
        self.dataset.y_train = min['y_train']
        #self.model = min['model']
        print("Saving best dataset")
        self.dataset.save(save_tokenizer=False)
        #self.model.save_model()
        #print(f"\n\nSaved model weight: {self.model.model.layers[1].get_weights()[0][0][0]}\nModel saving complete")


                




    def clear_weights(self):
        self.model.__init__()






