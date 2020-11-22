from sklearn.linear_model import LogisticRegression
from typing import Union

from ...core.model import AbstractModel
from ...core.data import Data
from ...core.pipe import Pipe


class ModelLogisticRegression(AbstractModel):
    __name__ = 'ModelLogisticRegression'

    def __init__(self, *, random_state: int = 42):
        super().__init__()
        self.random_state = random_state

    def fit(self, son: Pipe, X: Data, y: Data, params: Union[dict, None]):
        clf = LogisticRegression(random_state=self.random_state)
        clf.fit(X.train, y.train)
        self.model = clf

    def predict(self, X):
        return self.model.predict(X)
