import numpy as np
from maav.configuration import config
import maav.configuration as configuration
import tensorflow as tf
import copy
from collections import Counter
import random

class History:
    """
    Class to mimic tensorflow's training history instance
    """
    def __init__(self) -> None:
        self.history = {"loss":[], "val_loss":None, "loss_non_batched":[]}
    def is_loss_oscillating(self):
        try:
            last_5_loss_values = self.history["loss_non_batched"][-5:]
            element_counts = Counter(last_5_loss_values)
            for loss, number_of_occurence in element_counts.items():
                if number_of_occurence > 2:
                    config["MAX_UPDATE_FACTOR"] -= 0.0001
                    config["MIN_UPDATE_FACTOR"] += 0.0001
                    return True
            return False
        except IndexError:
            return False



class Weight_Manipulator:
    """
    This class is for manipulating weights of tensorflow model.
    Weight_Manipulator is an abstract layer for handling tensorflow model weights

    Methos:
        generate_random_grad(): To create a random grad with respect to weights and biases
        get_params_array(): Function to return a continous 1D array containing all weights of model
            _format_params(): reccuresive method used by get_params_array()
        apply_grad(): method to apply given grad to a given model parameters
            _apply_grad_to_weights(): reccrsive method used by apply_grad()
        calculate_gradient(): method to calculate grad given previous and new parameters

    """
    def __init__(self) -> None:
        self.shape_reference = None
        self.weights_count = 0
        self.line_params = []
        self.grad = []
        self.grad_array_index = 0
    def generate_random_grad(self):
        self.grad = []
        for _ in range(self.weights_count):
            self.grad.append(np.random.uniform(config["MIN_UPDATE_FACTOR"], config["MAX_UPDATE_FACTOR"]))
        return self.grad.copy()
    def _format_params(self, params):
        
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self._format_params(params[i])
            return params
        else:
            self.weights_count += 1
            self.line_params.append(params)
            return params
    
    def get_params_array(self, params):
        self.weights_count = 0
        self.line_params = []
        self.shape_reference = params
        self._format_params(params)
        return self.line_params


    
    def _apply_grad_to_weights(self, params):
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self._apply_grad_to_weights(params[i])
            return params
        else:
            params += self.grad[self.grad_array_index]
            self.grad_array_index += 1
            return params
    
    def apply_grad(self, grad, params):
        self.grad_array_index = 0
        self.grad = grad
        if len(grad) != self.weights_count:
            print("Shape error")
        else:
            return self._apply_grad_to_weights(params)
    def calculate_gradient(self, prev_params=[], new_params=[], negate=False):
        prev = self.get_params_array(prev_params)
        new = self.get_params_array(new_params)

        self.grad = []

        if len(prev) != len(new):
            print("Shape error")
        else:
            for index in range(len(prev)):
                self.grad.append(new[index] - prev[index])
                if negate:
                    self.grad[-1] = -self.grad[-1]
            return self.grad.copy()

class Ran_Optimizer:
    """
    The main optimizer implemention class
    This class uses ran optimization algorithm to optimize model weights to minimize loss function

    Methods:
        get_loss(): returns loss of model on train dataset
        train(): method to train model, given number of epochs
        step_update(): reccursive method used to train model using ran optimization algorithm


    """
    def __init__(self, model, x_train, y_train, epochs) -> None:
        self.model = model
        self.epochs = epochs
        self.x_train = x_train
        self.y_train = y_train
        self.prev_params = None
        self.prev_loss = None
        self.prev_accuracy = None
        self.weight_maniputlator = Weight_Manipulator()
        self.prev_params = None
        self.prev_loss = None
        self.new_params = None
        self.new_loss = None
        self.epoch_count = 0
        self.epochs = 0
        self.history = History()
    
    def roll_dataset(self):
        x_train = self.x_train[:config["BATCH_SIZE"]]
        y_train = self.y_train[:config["BATCH_SIZE"]]

        self.x_train = self.x_train[config["BATCH_SIZE"]:]
        self.y_train = self.y_train[config["BATCH_SIZE"]:]

        self.x_train = self.x_train + x_train
        self.y_train = self.y_train + y_train

    
    def shuffle(self):# not needed
        self.x_train, self.y_train = zip(*(random.sample(list(zip(self.x_train, self.y_train)), len(self.x_train))))

    def get_loss(self):
        x_test_tensor = tf.convert_to_tensor(self.x_train, dtype=tf.int32)
        y_test_tensor = tf.convert_to_tensor(self.y_train, dtype=tf.int32)
        something = self.model.evaluate(x_test_tensor, y_test_tensor)
        return something

    def get_loss_batched(self):
        #self.shuffle()
        self.roll_dataset()
        x_test_tensor = tf.convert_to_tensor(self.x_train[:100], dtype=tf.int32)
        y_test_tensor = tf.convert_to_tensor(self.y_train[:100], dtype=tf.int32)
        something = self.model.evaluate(x_test_tensor, y_test_tensor)
        return something

    def step_update(self, grad):
        #configuration.progress_bar("Training using adam", self.epoch_count, self.epochs)
        print(f"Ran: Epoch {self.epoch_count}/{self.epochs}")
        if self.epoch_count > self.epochs:
            return
        self.new_params = self.weight_maniputlator.apply_grad(grad, copy.deepcopy(self.prev_params))
        self.model.set_weights(self.new_params)
        self.new_loss = self.get_loss()
        #self.new_loss = np.random.uniform(-1, 1)
        self.history.history["loss"].append(self.new_loss)
        self.history.history["loss_non_batched"].append(self.new_loss)
        self.epoch_count += 1
        
        if self.new_loss < self.prev_loss:
            # model improved
            if self.history.is_loss_oscillating():
                new_grad = self.weight_maniputlator.generate_random_grad()
                print("generating another random grad")
            else:
                new_grad = self.weight_maniputlator.calculate_gradient(copy.deepcopy(self.prev_params), copy.deepcopy(self.new_params), negate=False)
            self.prev_params = self.new_params
            self.prev_loss = self.new_loss

            self.step_update(new_grad)
        else:
            # model not improved (negate gradient)
            if self.history.is_loss_oscillating():
                new_grad = self.weight_maniputlator.generate_random_grad()
                print("generatign another random grad")
            else:
                new_grad = self.weight_maniputlator.calculate_gradient(copy.deepcopy(self.prev_params), copy.deepcopy(self.new_params), negate=True)

            self.prev_params = self.new_params
            self.prev_loss = self.new_loss

            self.step_update(new_grad)
        return


    def train(self, epochs):
        self.model.build(input_shape=(None, config["MAX_SEQUENCE_LENGTH"]))
        
        self.epoch_count = 0
        self.epochs = epochs
        self.prev_params = self.model.get_weights()
        self.prev_loss = self.get_loss_batched()
        self.history.history["loss"].append(self.prev_loss)

        waste = self.weight_maniputlator.get_params_array(copy.deepcopy(self.prev_params))# called to give self.weight_manipulator a reference of the model parameter array structure
        grad = self.weight_maniputlator.generate_random_grad()
        self.step_update(grad)
        

        return self.history
        