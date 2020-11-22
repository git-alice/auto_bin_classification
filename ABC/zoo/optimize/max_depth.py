from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import Union
from sklearn.model_selection import train_test_split


from ABC.core.model import AbstractModel
from ABC.core.data import Data
from ABC.core.pipe import Pipe


class GridOptimizeMaxDepth(AbstractModel):
    __name__ = 'GridOptimizeMaxDepth'
    verbose = True

    def __init__(self):
        super().__init__()
        self.sons = []
        self.gridsearch_params = [max_depth for max_depth in range(2, 12)]
        self.best_son = None
        self.best_params = None

    def fit(self, son: Pipe, X: Data, y: Data, params: Union[dict, None]):
        min_accuracy_score = float("Inf")
        params = params if params else {}

        if X.validation.empty and y.validation.empty:
            X_train, X_validation, y_train, y_validation = train_test_split(X.train, y.train,
                                                                            test_size=0.1, random_state=42)
            X = Data(train=X_train, validation=X_validation)
            y = Data(train=y_train, validation=y_validation)

        for max_depth in self.gridsearch_params:
            _son = deepcopy(son)

            if self.verbose:
                print(f"Параметры: max_depth={max_depth}")

            params.update({'max_depth': max_depth})
            _son.fit(X, y, params)
            self.sons.append(son)

            current_accuracy_score = accuracy_score(y.validation, _son.predict(X.validation))

            if self.verbose:
                print(f"\tAccuracy: {current_accuracy_score}")

            if current_accuracy_score < min_accuracy_score:
                min_accuracy_score = current_accuracy_score
                self.best_params = dict(max_depth=max_depth)
                self.best_son = _son

        if self.verbose:
            print(f"Лучшие параметры:\n\tmax_depth: {self.best_params['max_depth']}")

    def predict(self, X):
        return self.best_son.predict(X)
