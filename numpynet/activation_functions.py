import numpy as np
from abc import abstractmethod, ABC
from .util import check_2d_array

class ActivationFunction(ABC):
    @abstractmethod
    def _validate_input_shape(self, input):
        pass

    @abstractmethod
    def apply(self, input):
        self._validate_input_shape(input)
        check_2d_array(input)

    @abstractmethod
    def grad(self, input):
        self._validate_input_shape(input)
        check_2d_array(input)


class Tanh(ActivationFunction):
    def _validate_input_shape(self, input):
        pass

    def apply(self, input):
        super(Tanh, self).apply(input)
        return np.tanh(input)

    def grad(self, input):
        super(Tanh, self).grad(input)
        return 1 - np.square(np.tanh(input))


class Lin(ActivationFunction):
    def apply(self, input):
        super(Lin, self).apply(input)

        return input

    def grad(self, input):
        super(Lin, self).grad(input)

        return np.repeat(
            np.identity(input.shape[1])[None,...],
            input.shape[0],
            axis=0
        )


class Softmax(ActivationFunction):
    def _validate_input_shape(self, input):
        assert(input.ndim == 2)

    def apply(self, input):
        super(Softmax, self).apply(input)

        exponents = np.exp(input)
        normalization = np.apply_over_axes(np.sum, exponents, axes=(-1,))
        return exponents / normalization

    def grad(self, input):
        super(Softmax, self).grad(input)

        res = self.apply(input)
        outer = res[...,None] * res[:,None,:]
        diag = np.apply_along_axis(np.diag, 1, res)
        return diag - outer