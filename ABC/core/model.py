from pandas import DataFrame
from abc import abstractmethod

from .data import Data
from .pipe import Pipe


class AbstractModel:
    __type__ = 'model'

    def __init__(self):
        self.model = None
        self.son = None

    @abstractmethod
    def fit(self, son: Pipe, X: Data, y: Data, params: dict):
        pass

    @abstractmethod
    def predict(self, X: DataFrame):
        pass
