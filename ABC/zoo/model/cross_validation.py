from copy import deepcopy
from sklearn.model_selection import KFold
import pandas as pd
from typing import Union

from ABC.core.model import AbstractModel
from ABC.core.data import Data
from ABC.core.pipe import Pipe


class ModelCrossValidation(AbstractModel):
    __name__ = 'ModelCrossValidation'

    def __init__(self, ):
        super().__init__()
        self.sons = []

    def fit(self, son: Pipe, X: Data, y: Data, params: Union[dict, None]):
        kf = KFold(n_splits=3)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X.train):
            son = deepcopy(son)

            X_train, X_validation = X.train.iloc[train_index], X.train.iloc[test_index]
            y_train, y_validation = y.train.iloc[train_index], y.train.iloc[test_index]

            son.fit(Data(X_train, X_validation), Data(y_train, y_validation))
            self.sons.append(son)

    def predict(self, X):
        predictions = pd.DataFrame()

        for i, son in enumerate(self.sons):
            prediction = son.predict(X)
            predictions.loc[:, f'validation_on_fold_{i}'] = prediction

        predictions.loc[:, 'mode'] = predictions.mode(axis=1).values.reshape(1, -1)[0]
        return predictions['mode']
