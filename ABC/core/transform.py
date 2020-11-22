from pandas import DataFrame
from abc import abstractmethod

from .pipe import Pipe
from .data import Data


class AbstractTransform:
    __type__ = 'transform'

    def __init__(self):
        self.transformer = None
        self.son = None

    @abstractmethod
    def transform(self, son: Pipe, X: Data, y: Data):
        pass

    @abstractmethod
    def fit(self, son: Pipe, X: Data, y: Data):
        pass

    @abstractmethod
    def predict(self, X: DataFrame):
        pass
