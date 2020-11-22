from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import Union

from ABC.core.model import AbstractModel
from ABC.core.data import Data
from ABC.core.pipe import Pipe


class GridOptimizeMinChildWeight(AbstractModel):
    __name__ = 'GridOptimizeMinChildWeight'
    verbose = True

    def __init__(self):
        super().__init__()
        self.sons = []
        self.gridsearch_params = [min_child_weight for min_child_weight in range(2, 8)]
        self.best_son = None
        self.best_params = None

    def fit(self, son: Pipe, X: Data, y: Data, params: Union[dict, None]):
        min_accuracy_score = float("Inf")
        params = params if params else {}

        if not X.validation.empty and not y.validation.empty:
            X_train, X_validation, y_train, y_validation = train_test_split(X.train, y.train,
                                                                            test_size=0.1, random_state=42)
            X = Data(train=X_train, validation=X_validation)
            y = Data(train=y_train, validation=y_validation)

        for min_child_weight in self.gridsearch_params:
            _son = deepcopy(son)

            if self.verbose:
                print(f"Параметры: min_child_weight={min_child_weight}")

            params.update({'min_child_weight': min_child_weight})
            _son.fit(X, y, params)
            self.sons.append(_son)

            current_accuracy_score = accuracy_score(y.validation, _son.predict(X.validation))

            if self.verbose:
                print(f"\tAccuracy: {current_accuracy_score}")

            if current_accuracy_score < min_accuracy_score:
                min_accuracy_score = current_accuracy_score
                self.best_params = dict(min_child_weight=min_child_weight)
                self.best_son = _son

        if self.verbose:
            print(f"Лучшие параметры:\n\tmin_child_weight: {self.best_params['min_child_weight']}")

    def predict(self, X):
        return self.best_son.predict(X)
