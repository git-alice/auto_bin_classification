from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report


class MetricEvaluator:
    def __init__(self, model, *, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def get_report(self):
        return classification_report(self.y_test, self.model.predict(self.X_test))

    def get_precision_score(self):
        return precision_score(self.y_test, self.model.predict(self.X_test), average='weighted')

    def get_accuracy_score(self):
        return accuracy_score(self.y_test, self.model.predict(self.X_test))

    def get_f1_score(self):
        return f1_score(self.y_test, self.model.predict(self.X_test))
