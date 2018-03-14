import numpy as np
from .activation_functions import Tanh

class Dense(object):
    def __init__(
        self,
        input_size,
        size,
        activation_function=Tanh()
    ):
        self.input_size = input_size
        self.size = size
        self.activation_function=activation_function

        self._initalize()

    def _initalize(self):
        self.W = np.random.randn(
            self.size,
            self.input_size
        ) - 0.5

        self.b = np.random.randn(self.size) - 0.5

    def _stimuli(self, X):
        reshaped = np.reshape(X, (-1, self.input_size))
        return np.tensordot(reshaped, self.W, (1, 1)) \
               + self.b[np.newaxis, :].repeat(X.shape[0], axis=0)

    def forward(self, X):
        input = self._stimuli(X)
        return self.activation_function.apply(input), input

    def backward(
            self,
            prev_layer_state,
            next_layer_errors,
            stimuli = None
    ):
        stimuli = None
        if type(stimuli) is not np.ndarray:
            stimuli = self._stimuli(prev_layer_state)

        # Number of samples passed
        n = stimuli.shape[0]

        activ_func_grad = self.activation_function.grad(
            stimuli
        )

        if activ_func_grad.ndim == 3:
            stimulus_error = np.einsum('...ij,...j', activ_func_grad, next_layer_errors)
        else:
            stimulus_error = activ_func_grad * next_layer_errors

        grad_b = stimulus_error

        errors = np.einsum('...ji,...j', self.W, stimulus_error)

        prev_layer_state = prev_layer_state.repeat(self.size, axis=0).reshape(-1, self.size, self.input_size)
        stimulus_error = stimulus_error.T.repeat(self.input_size,axis=1).reshape(-1, self.size, self.input_size)

        grad_w = (prev_layer_state * stimulus_error).reshape(-1, self.size, self.input_size)

        return grad_w, grad_b, errors