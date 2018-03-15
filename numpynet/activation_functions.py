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

    @abstractmethod
    def grad(self, input):
        self._validate_input_shape(input)


class Tanh(ActivationFunction):
    def _validate_input_shape(self, input):
        pass

    def apply(self, input):
        super(Tanh, self).apply(input)
        return np.tanh(input)

    def grad(self, input):
        super(Tanh, self).grad(input)
        return 1 - np.square(np.tanh(input))


class ReLu(ActivationFunction):
    def _validate_input_shape(self, input):
        pass

    def apply(self, input):
        def apply(self, input):
            super(ReLu, self).apply(input)
        return np.maximum(input, 0)

    def grad(self, input):
        super(ReLu, self).grad(input)
        return 1.0 * (input > 0)


class Lin(ActivationFunction):
    def _validate_input_shape(self, input):
        check_2d_array(input)

    def apply(self, input):
        super(Lin, self).apply(input)

        return input

    def grad(self, input):
        super(Lin, self).grad(input)

        # TODO: this now only supports 2d inputs....
        return np.repeat(
            np.identity(input.shape[1])[None,...],
            input.shape[0],
            axis=0
        )


class Softmax(ActivationFunction):
    def _validate_input_shape(self, input):
        check_2d_array(input)

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