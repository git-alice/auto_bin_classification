from datetime import datetime


class Pickled:
    def __init__(self, item):
        self.item = item
        self.time: datetime = datetime.now()
        self.description: str = ''

    def save_data_hook(self, **kwargs: dict) -> None:
        self.time = datetime.now()
        self.description: str = kwargs.get('description', 'отсутствует')

    def load_data_hook(self):
        print(f'Объект был сохранен: {self.time}')
        print(f'Описание: {self.description}')
