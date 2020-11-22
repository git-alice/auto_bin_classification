import cloudpickle as pickle
from pathlib import Path
from typing import Dict, Any, List, Union

from ABC.core.storage.pickled import Pickled


class Storage:
    root: Path = Path('./data/')
    verbose: bool = True
    _verbose_print_margin: str = '\t'

    @classmethod
    def set_root(cls, root: Union[str, Path]) -> None:
        cls.root = Path(root)

    @classmethod
    def save(cls, item: Any, filename: Union[str, Path], sub_storage: Union[str, Path] = '', **kwargs: Dict[str, Any]) -> None:
        print('Сохранение:')

        filename = Path(filename)

        if cls.verbose:
            print(f'{cls._verbose_print_margin}Текущая директория: {Path.cwd()}')
            print(f'{cls._verbose_print_margin}Корень базы данных: {cls.root}')
            print(f'{cls._verbose_print_margin}Хранилище/Подхранилище: {sub_storage or cls.root}')

        cls.root.mkdir(parents=True, exist_ok=True)

        if not filename.is_absolute():
            (cls.root / sub_storage / filename).parent.mkdir(parents=True, exist_ok=True)

        try:
            filename = Path(f'{filename}.pickle')
            pickled_item = Pickled(item=item)
            pickled_item.save_data_hook(**kwargs)
            with open(cls.root / sub_storage / filename, 'wb') as f:
                pickle.dump(pickled_item, f)
        except Exception as e:
            print(f'{cls._verbose_print_margin}Тип данных не соответвует базе данных.')
            print(f'{cls._verbose_print_margin}Тип объекта: ', type(item))
            print(f'{cls._verbose_print_margin}Исключение:', e)

    @classmethod
    def load(cls, filename: str, sub_storage: Union[str, Path] = '') -> Pickled:
        print('Загрузка:')

        try:
            filename = Path(f'{filename}.pickle')
            print(f'{cls._verbose_print_margin}Загрузка из: {cls.root / sub_storage / filename}') if cls.verbose else None
            with open(cls.root / sub_storage / filename, 'rb') as f:
                pickled = pickle.load(f)
            pickled.load_data_hook()
            return pickled.item
        except FileNotFoundError:
            print(f'{cls._verbose_print_margin}Объект отсутствует в базе данных.')

    @classmethod
    def get_storages(cls) -> List[str]:
        return [p.stem for p in cls.root.glob('*') if p.is_dir()]

    @classmethod
    def get_all_names_from_storage(cls, storage: Union[str, Path] = '', ext: str = 'pickle') -> List[str]:
        return [f.stem for f in cls.root.glob(f'{storage}/*.{ext}') if f.is_file()]

    @classmethod
    def load_all_from_storage(cls, storage: Union[str, Path] = '') -> List[Pickled]:
        objs = [cls.load(f) for f in cls.get_all_names_from_storage(storage)]
        return objs
