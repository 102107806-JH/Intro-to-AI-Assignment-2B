import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class ToTensor:
    def __call__(self, datum):
        x, y = datum
        return torch.from_numpy(x), torch.from_numpy(y)

class ScaleAndShiftX:
    def __init__(self, feature_index, divisor):
        self._feature_index = feature_index
        self._divisor = divisor

    def __call__(self, datum):
        x, y = datum
        x[:, self._feature_index] -= (self._divisor / 2)
        x[:, self._feature_index] /= (self._divisor / 2)
        return x, y

class ScaleY:
    def __init__(self, divisor, minAfterDiv=0, maxAfterDiv=1):
        self._divisor = divisor
        self._dif = maxAfterDiv - minAfterDiv

    def __call__(self, datum):
        x, y = datum
        y /= self._divisor
        y *= self._dif
        y += self._dif / 2
        return x, y



