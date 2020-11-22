from ABC.core.pipe import Pipe
from ABC.core.data import Data
from ABC.zoo.optimize.max_depth import GridOptimizeMaxDepth
from ABC.zoo.optimize.min_child_weight import GridOptimizeMinChildWeight
from ABC.zoo.model.xgb_classifier import ModelXGB
from ABC.zoo.model.decision_tree import ModelDecisionTree
from ABC.zoo.model.cross_validation import ModelCrossValidation
from ABC.zoo.transform.standard_scaler import TransformStandardScaler
from ABC.zoo.model.logistic_regression import ModelLogisticRegression
from ABC.zoo.model.svc import ModelSVC


my_xgb = Pipe([
    ModelCrossValidation(),
    GridOptimizeMaxDepth(),
    GridOptimizeMinChildWeight(),
    ModelXGB()
])

my_dt = Pipe([
    GridOptimizeMaxDepth(),
    ModelDecisionTree()
])

my_svc = Pipe([
    ModelSVC()
])

my_lr = Pipe([
    TransformStandardScaler(),
    ModelLogisticRegression()
])

animals = [my_xgb, my_dt, my_svc, my_lr]
