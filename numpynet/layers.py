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

class Conv2d(object):
    """
    Convolutional layer that performs 2d convolution.
    This layer highly relies on the im2col and col2im
    implementations provided by the Stanford CS231 class.
    """
    def __init__(
        self,
        input_size,
        filter_size=(2,2),
        n_filters=10,
        padding=1,
        stride=1,
        activation_function=Tanh()
    ):
        self.n_channels = input_size[0]
        self.width = input_size[1]
        self.height = input_size[2]
        self.filter_width = filter_size[0]
        self.filter_height = filter_size[1]
        self.n_filters = n_filters
        self.padding = padding
        self.stride = stride
        self.activation_function = activation_function

        self._initialize()

    def _initialize(self):
        """Initialize the weights for this layer."""
        self.W = np.random.randn(
            self.n_filters,
            self.filter_width * self.filter_height * self.n_channels
        )
        self.b = np.random.randn(
            self.n_filters
        )

    def _stimuli(self, X):
        res = np.matmul(self.W, self.im2col(X))

        # Add the constant
        res = res + self.b[:,np.newaxis].repeat(res.shape[1], axis=1)

        # Reshape the result to images again
        res = res.reshape(
            self.n_filters,
            int((X.shape[2] + 2 * self.padding - self.filter_height) / self.stride + 1),
            int((X.shape[3] + 2 * self.padding - self.filter_width) / self.stride + 1),
            X.shape[0]
        ).transpose(3, 0, 1, 2)


        return res

    def get_im2col_indices(self, x_shape):
        # Get parameters for the transformation
        field_height = self.filter_height
        field_width = self.filter_width
        padding = self.padding
        stride = self.stride

        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_height) % stride == 0
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    def im2col(self, x):
        # Get parameters for the transformation
        field_height = self.filter_height
        field_width = self.filter_width
        padding = self.padding

        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def _validate_input_shape(self, X):
        assert X.shape[1:] == (self.n_channels, self.width, self.height)

    def forward(self, X):
        self._validate_input_shape(X)

        inputs = self._stimuli(X)

        return self.activation_function.apply(inputs), inputs

    def backward(
        self,
        prev_layer_state,
        next_layer_errors,
        stimuli=None
    ):
        if type(stimuli) is not np.ndarray:
            stimuli = self._stimuli(prev_layer_state)

        activ_grad = self.activation_function.grad(stimuli)
        stimulus_grad = next_layer_errors * activ_grad

        grad_b = np.sum(stimulus_grad, axis=(0,2,3))



        #TODO: calculate grad_W, calculate_errors

        return None, grad_b, None


class Flatten(object):
    def __init__(
        self,
        input_shape
    ):
        self.input_shape = input_shape

        #TODO: temp solution:
        self.W = 0
        self.b = 0


    @property
    def size(self):
        size=1
        for i in self.input_shape:
            size = size*i
        return size

    def forward(self, X):
        return X.reshape(-1, self.size), X


    def backward(
        self,
        prev_layer_state,
        next_layer_errors,
        stimuli=None
    ):
        return 0, 0, next_layer_errors.reshape((-1,) + self.input_shape)