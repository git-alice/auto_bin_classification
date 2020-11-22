from loguru import logger
from copy import deepcopy


class Pipe:
    def __init__(self, init_pipeline_list):
        self.init_pipeline_list = init_pipeline_list
        self.father = None

    def predict(self, X):
        prediction = self.father.predict(X)
        return prediction

    def fit(self, X, y, params=None):
        father = self.init_pipeline_list[0]
        son = Pipe(self.init_pipeline_list[1:])
        father.fit(son, X, y, params)

        self.father = father

    def __repr__(self):
        return self.__print__()

    def __print__(self, depth=0):
        tab = ' ' * 2
        new_line = '\n'
        if self.father:
            res = f"{self.father.__name__}: {id(self.father)}{new_line}{tab * (depth + 1)}"
        else:
            raise Exception('Пайплайн еще не обучен')

        if hasattr(self.father, 'son'):
            if self.father.son is None:
                res += f"└End"
            else:
                res += f"└{self.father.son.__print__(depth + 1)}{new_line}{tab * (depth + 1)}"
        elif hasattr(self.father, 'sons'):
            for son in self.father.sons:
                res += f"└{son.__print__(depth + 1)}{new_line}{tab * (depth + 1)}"
        else:
            raise Exception('Неправильная структура пайплайна.')

        return res
