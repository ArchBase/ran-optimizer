from maav.configuration import config
import maav.configuration as configuration

class optimizer:
    def __init__(self, model, x_train, y_train) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train(self, epochs=0):
        for epoch in range(epochs):
            configuration.progress_bar("training ran", epoch, epochs)
            prev_params = self.model.get_weights()
        print(prev_params)

