import numpy as np
from maav.configuration import config
import maav.configuration as configuration
import tensorflow as tf

class Weight_Manipulator:
    def __init__(self) -> None:
        self.shape_reference = None
        self.weights_count
        self.line_params = []
    def _format_params(self, params):
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self._format_params(params[i])
            return params
        else:
            self.line_params.append(params)
            return params
    
    def get_params_array(self, params):
        self._format_params(params)
        return self.line_params

    def _ran_update_params(self, params):
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self._ran_update_params(params[i])
            return params
        else:
            params += np.random.uniform(config["MIN_UPDATE_FACTOR"], config["MAX_UPDATE_FACTOR"])
            return params
        
    def random_update_params(self, params):
        self.shape_reference = params
        return self._ran_update_params(params)

class optimizer:
    def __init__(self, model, x_train, y_train) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.prev_params = None
        self.prev_loss = None
        self.prev_accuracy = None
        self.weight_maniputlator = Weight_Manipulator()
    
    def calculate_gradient(self, prev=[], new=[]):
        prev = self.weight_maniputlator.get_params_array(prev)
        new = self.weight_maniputlator.get_params_array(new)

        grad = []

        if len(prev) != len(new):
            print("Shape error")
        else:
            for index in range(len(prev)):
                grad.append(new[index] - prev[index])
        return grad
    
    def apply_grad(self, grad=[]):


    def get_loss(self):
        x_test_tensor = tf.convert_to_tensor(self.x_train, dtype=tf.int32)
        y_test_tensor = tf.convert_to_tensor(self.y_train, dtype=tf.int32)
        something = self.model.evaluate(x_test_tensor, y_test_tensor)
        return something

    def train(self, epochs=0):
        self.model.build(input_shape=(None, config["MAX_SEQUENCE_LENGTH"]))
        #for epoch in range(epochs):
        #configuration.progress_bar("Training ran", epoch, epochs)
        #self.prev_params = self.model.get_weights()
        print(self.prev_params)
        p = self.get_loss()
        print("\nold\n")
        print(p)
        print("\n\n\nnew\n\n\n")
        s = self.weight_maniputlator.random_update_params(self.model.get_weights())
        #print(s)
        self.model.set_weights(s)
        print(self.get_loss())
        #self.prev_loss, 


        return
        self.model.build(input_shape=(None, config["MAX_SEQUENCE_LENGTH"]))
        for epoch in range(epochs):
            configuration.progress_bar("training ran", epoch, epochs)
        prev_params = self.model.get_weights()
        print(prev_params)
        self.random_update_params(prev_params)
        print("\n\n\n\nnew modifiedn\n\n\n\n")
        print(prev_params)

        #perturbation = np.random.normal(loc=0, scale=0.01, size=prev_params.shape)
        #modified_weights = prev_params + perturbation
        #print(prev_params)

