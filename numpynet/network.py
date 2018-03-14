import numpy as np
from .loss_functions import SquaredLoss

class Network(object):
    def __init__(
            self,
            loss_function=SquaredLoss()
    ):
        self.layers = []
        self.loss_function = loss_function

    def add(self, layer):
        self.layers.append(layer)

    def _apply(self, X):
        states = [X]
        stimuli = [X]

        for layer in self.layers:
            state, stimulus = layer.forward(states[-1])
            states.append(state)
            stimuli.append(stimulus)

        return states, stimuli

    def predict(self, X):
        states, stimuli = self._apply(X)
        return states[-1]

    def fit_batch(self, X, y, learning_rate=0.1):
        states, stimuli = self._apply(X)

        error = self.loss_function.grad(states[-1], y)

        layers = reversed(self.layers)
        states = reversed(states[:-1])
        stimuli = reversed(stimuli[1:])

        for layer, prev_layer_state, stimulus in zip(layers, states, stimuli):
            grad_W, grad_b, error = layer.backward(prev_layer_state, error, stimulus)

            layer.W = layer.W - learning_rate * grad_W.mean(axis=0)
            layer.b = layer.b - learning_rate * grad_b.mean(axis=(0,1))

    def fit(self, X, y, batch_size=10, epochs=100, learning_rate=0.1):
        for i in range(epochs):
            ind = np.random.choice(np.arange(X.shape[0]), batch_size, replace=False)
            self.fit_batch(X[ind], y[ind], learning_rate)

            y_pred = self.predict(X)

            print('Current loss {0}'.format(
                self.loss_function.compute(y_pred, y)
            ))



