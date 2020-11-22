import xgboost as xgb
from typing import Union

from ...core.model import AbstractModel


class ModelXGB(AbstractModel):
    __name__ = 'ModelXGB'

    def __init__(self, num_boost_round: int = 999):
        super().__init__()
        self.params = {
            'max_depth': 6,
            'min_child_weight': 1,
            'eta': .3,
            'subsample': 1,
            'colsample_bytree': 1,
            'objective': 'binary:logistic'
        }
        self.random_state = 420

    def fit(self, son, X, y, params: Union[dict, None]):
        params = params if params else {}

        self.params.update(params)
        clf = xgb.XGBClassifier(**self.params, n_estimators=100, random_state=self.random_state)
        clf.fit(X.train, y.train)
        self.model = clf

    def predict(self, X):
        prediction = self.model.predict(X)
        return prediction
