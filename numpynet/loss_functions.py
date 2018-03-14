import numpy as np

class SquaredLoss(object):
    def compute(self, input, target):
        return np.square(input-target).mean()

    def grad(self, input, target):
        return 2.0*(input-target)

class CrossEntropy(object):
    def compute(self, input, target):
        return -(np.multiply(target, np.log(input))).sum(axis=1).mean()

    def grad(self, input, target):
        return - np.multiply(target, (1/input))