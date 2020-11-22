from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import uuid
from pathlib import Path
import sys
from copy import deepcopy

from .data import Data
from .metric import MetricEvaluator
from .storage import Storage

from ..zoo.animals import animals as raw_animals


class AutoML:
    def __init__(self):
        self.raw_animals = deepcopy(raw_animals)
        self.animals = []
        self.my_id = uuid.uuid4().hex

        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None

    def fit_all_animals(self, X: pd.DataFrame, y: pd.DataFrame, *, parallel: bool = True, n_jobs: int = 4,
                        stdout_to_file: bool = True, test_size: float = 0.3, random_state: int = 42):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        X_data = Data(train=X_train, validation=pd.DataFrame())
        y_data = Data(train=y_train, validation=pd.Series())

        # Читать вывод при параллельном вычислении бессмылсенно
        if stdout_to_file:
            original_stdout = sys.stdout
            f = open('log.txt', 'w')
            sys.stdout = f

        if parallel:
            Parallel(n_jobs=n_jobs)(delayed(animal.fit)(X_data, y_data) for animal in raw_animals)

            self.animals = list(map(lambda x: MetricEvaluator(x, X_test=X_test, y_test=y_test), raw_animals))

        else:
            for raw_animal in raw_animals:
                raw_animal.fit(X_data, y_data)
                self.animals.append(MetricEvaluator(raw_animal, X_test=X_test, y_test=y_test))

        if stdout_to_file:
            sys.stdout = original_stdout
            f.close()

    def report_all_animals(self):
        for animal in self.animals:
            print('.' * 15)
            print(animal.model)
            print(animal.get_report())

    def print_animals(self):
        print('Обученные модели:\n')
        for a in self.animals:
            print(a.model)
            print()

    def get_best_animal(self):
        scores = list(map(lambda x: x.get_accuracy_score(), self.animals))
        _index = int(np.argmax(scores))
        return self.animals[_index].model

    def save_best_animal(self):
        best_animal = self.get_best_animal()
        Storage.save(best_animal, filename='model', sub_storage=Path(self.my_id, 'best_animal'))

    def load_best_animal(self):
        return Storage.load(filename='model', sub_storage=Path(self.my_id, 'best_animal'))
