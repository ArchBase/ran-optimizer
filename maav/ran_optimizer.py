import numpy as np
from maav.configuration import config
import maav.configuration as configuration
import tensorflow as tf
import copy

class Weight_Manipulator:
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

    def _ran_update_params(self, params):
        self.weights_count += 1
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self._ran_update_params(params[i])
            return params
        else:
            params += np.random.uniform(config["MIN_UPDATE_FACTOR"], config["MAX_UPDATE_FACTOR"])
            return params
        
    def random_update_params(self, params):
        self.weights_count = 0
        self.shape_reference = params
        return self._ran_update_params(params)
    
    def _apply_grad_to_weights(self, params):
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self._apply_grad_to_weights(params[i])
            return params
        else:
            #print("(((((())))))" + str(self.grad_array_index))
            #print(self.grad)
            params += self.grad[self.grad_array_index]
            self.grad_array_index += 1
            #print("updating")
            return params
    
    def apply_grad(self, grad, params):
        self.grad_array_index = 0
        self.grad = grad
        if len(grad) != self.weights_count:
            print("Shape error")
        else:
            return self._apply_grad_to_weights(params)
    def calculate_gradient(self, prev_params=[], new_params=[], negate=False):
        #print("Params: ")
        #print(prev_params)
        #print(new_params)
        prev = self.get_params_array(prev_params)
        new = self.get_params_array(new_params)

        self.grad = []

        if len(prev) != len(new):
            print("Shape error")
        else:
            for index in range(len(prev)):
                #print("updating " + str(new[index]) + " and " + str(prev[index]))
                self.grad.append(new[index] - prev[index])
                if negate:
                    self.grad[-1] = -self.grad[-1]
            return self.grad.copy()

class optimizer:
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
    
    

    def get_loss(self):
        x_test_tensor = tf.convert_to_tensor(self.x_train, dtype=tf.int32)
        y_test_tensor = tf.convert_to_tensor(self.y_train, dtype=tf.int32)
        something = self.model.evaluate(x_test_tensor, y_test_tensor)
        return something

    def step_update(self, grad):
        if self.epoch_count > self.epochs:
            return
        self.new_params = self.weight_maniputlator.apply_grad(grad, copy.deepcopy(self.prev_params))
        self.model.set_weights(self.new_params)
        self.new_loss = self.get_loss()
        self.epoch_count += 1
        if self.new_loss < self.prev_loss:
            # model improved
            new_grad = self.weight_maniputlator.calculate_gradient(copy.deepcopy(self.prev_params), copy.deepcopy(self.new_params), negate=False)

            self.prev_params = self.new_params
            self.prev_loss = self.new_loss

            self.step_update(new_grad)
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
        self.prev_loss = self.get_loss()

        waste = self.weight_maniputlator.get_params_array(copy.deepcopy(self.prev_params))
        grad = self.weight_maniputlator.generate_random_grad()
        #print("grad")
        #print(grad)
        self.step_update(grad)
        

        return
        self.prev_params = self.model.get_weights()
        self.prev_loss = self.get_loss()

        self.new_params = self.weight_maniputlator.random_update_params(self.prev_params)
        self.model.set_weights(self.new_params)
        self.new_loss = self.get_loss()

        if self.new_loss < self.prev_loss:
            grad = self.weight_maniputlator.calculate_gradient(self.prev_params, self.new_params, negate=False)
        else:
            grad = self.weight_maniputlator.calculate_gradient(self.prev_params, self.new_params, negate=True)
        



        return
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

