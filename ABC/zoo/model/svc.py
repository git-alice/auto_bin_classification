from sklearn.svm import SVC

from ...core.model import AbstractModel
from ...core.data import Data
from ...core.pipe import Pipe


class ModelSVC(AbstractModel):
    __name__ = 'ModelSVC'

    def __init__(self,):
        super().__init__()
        self.params = dict(kernel='linear')
        self.son = None

    def fit(self, son: Pipe, X: Data, y: Data, params: dict):
        params = params if params else {}

        self.params.update(params)
        clf = SVC(**self.params)
        clf.fit(X.train, y.train)
        self.model = clf

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction
