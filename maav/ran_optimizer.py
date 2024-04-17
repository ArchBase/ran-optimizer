import numpy as np
from maav.configuration import config
import maav.configuration as configuration

class Weigth_Manipulator:
    def __init__(self) -> None:
        self.shape_reference = None

    def _ran_update_params(self, params):
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self.random_update_params(params[i])
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
    def random_update_params(self, params):
        #print(params)
        #print(type(params))
        if isinstance(params, np.ndarray) or type(params) is list:
            for i in range(len(params)):
                params[i] = self.random_update_params(params[i])
            return params
        else:
            params += np.random.uniform(config["MIN_UPDATE_FACTOR"], config["MAX_UPDATE_FACTOR"])
            return params

        
        buffer = None
        for each in params:
            #print(type(each))
            if isinstance(each, np.ndarray):
                self.random_update_params(each)
                return
            else:
                each += np.random.uniform(config["MIN_UPDATE_FACTOR"], config["MAX_UPDATE_FACTOR"])
                buffer.append(each)
                
        return buffer
        
    def train(self, epochs=0):
        self.model.build(input_shape=(None, config["MAX_SEQUENCE_LENGTH"]))
        #for epoch in range(epochs):
        #configuration.progress_bar("Training ran", epoch, epochs)
        self.prev_params = self.model.get_weights()
        print(self.prev_params)
        print("\n\n\njjjjjjjjj\n\n\n")
        s = self.random_update_params(self.prev_params)
        print(s)
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

