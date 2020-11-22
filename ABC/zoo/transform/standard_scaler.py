from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from ...core.transform import AbstractTransform
from ...core.pipe import Pipe
from ...core.data import Data


class TransformStandardScaler(AbstractTransform):
    __name__ = 'TransformStandardScaler'

    def fit(self, son: Pipe, X: Data, y: Data, params: dict):
        self.son = son
        self.transformer = StandardScaler()
        X_new, y_new = deepcopy(X), deepcopy(y)
        X_new.train = self.transformer.fit_transform(X_new.train)
        son.fit(X_new, y_new, params)

    def predict(self, X):
        X_new = self.transformer.transform(X)
        return self.son.predict(X_new)
