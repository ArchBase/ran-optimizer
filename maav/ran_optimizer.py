import numpy as np
from maav.configuration import config
import maav.configuration as configuration

class optimizer:
    def __init__(self, model, x_train, y_train) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train(self, epochs=0):
        self.model.build(input_shape=(None, config["MAX_SEQUENCE_LENGTH"]))
        for epoch in range(epochs):
            configuration.progress_bar("training ran", epoch, epochs)
        prev_params = self.model.get_weights()
        #perturbation = np.random.normal(loc=0, scale=0.01, size=prev_params.shape)
        #modified_weights = prev_params + perturbation
        print(prev_params)

