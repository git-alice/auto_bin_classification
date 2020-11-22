********************************
Auto Binary Classification (ABC)
********************************


Мотивация:
---------
Существует много различных библиотек для создания моделей машинного обучения. Хочется иметь один интерфейс работы с ними, ведь в теории нужно использовать большинство из них. Поэтому иблиотека основана на абстрации `Pipe`, котороая позваоляет иметь одинаковых интерефейс для работы с различными моделямя, ведь все они обучаются и делают предсказания.

Структрура библиотеки:
---------------------
 - ABC
    - core
        - storage - отвечает за сохранения и загрузку, использует cloudpickle, так как не все объекты сохраняются с помощью встроенного
        - auto - отвечает за загрзку моделей и их обучение. На данный момент используются несколько моделей, логика которых написана в `ABC.zoo.animals`, далее все обучаются и выбирается лучшая. Но если моделей будет много, можно брать произвольно или делать модели второго уровня - библиотека это позволяет.
        - model, transform - два абстракнхы класса, который задают макет `Моделей` и `Трансформаций` над данными
        - pipe - обертка на `model` и `transform`
        - metric - отвечает за оценку модели (обертка)
    - zoo
        - model (optimize) - набор уже имплементированных моделей
        - transform - набор преобразователей данных (к примеру различные скейлеры или какое то добавление фичей можно делать тут)

Что можно у лучшить:
--------------------
 - Более консистетно использовать `Модели`, `Трансформы` и `Оптимизаторы`. К примеру, все они принимают метод `fit` объекты `Data`, а метод `predict` уже пандосовский датафрейм.
 - Сделать сохранение данных в более комактный вид (numpy array)
 - Сохранять нативными средствами, такими как joblib (Scikit) и save_model (CatBoost)
 - ModelCrossValidation усредняет предсказания голосованием, что является не лучшей идеей и стоит усредняться веротности
 - Скорее всего достатно багов, так что нужно тестировать
 - Много еще всего


Чтобы лучше понять как все рабоатет, посмотрите на ноутбуки в одноименной папке!