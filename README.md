# Проект по парсингу докуметнов

Этот проект использует модель `LayoutLMv3` из библиотеки Transformers от Hugging Face для классификации токенов на изображениях документов с использованием датасета FUNSD.

## Структура проекта

### data/
Директория, содержащая датасет FUNSD и предобработанные данные для модели LayoutLMv3.

### model/
- **checkpoint-1000/**
  - **config.json**: Конфигурационный файл модели, содержащий параметры модели.
  - **pytorch_model.bin**: Файл с весами обученной модели.
  - **tokenizer_config.json**: Конфигурационный файл токенизатора.
  - **vocab.txt**: Словарь токенизатора.
  - **special_tokens_map.json**: Файл со специальными токенами токенизатора.
  - **merges.txt**: Файл слияний BPE токенизатора.

### main/
- **train_model.py**: Скрипт для обучения модели LayoutLMv3 на датасете FUNSD. Основные шаги:
  - Загрузка датасета
  - Предобработка данных
  - Определение модели и параметров обучения
  - Запуск процесса обучения
  - Сохранение обученной модели и метрик
- **predict_model.py**: Скрипт для выполнения инференса с обученной моделью. Основные шаги:
  - Загрузка обученной модели и процессора
  - Загрузка изображения и его предобработка
  - Выполнение предсказания
  - Визуализация результатов предсказания
- **data_load.py**: Скрипт для загрузки и предобработки данных. Основные шаги:
  - Загрузка исходных данных
  - Преобразование данных в необходимый формат
  - Сохранение предобработанных данных для дальнейшего использования

### requirements.txt
Файл, содержащий список зависимостей Python, необходимых для запуска проекта. Примеры зависимостей:

## Начало работы

### Установка зависимостей

Для начала работы с проектом, убедитесь, что у вас установлены все необходимые зависимости. Вы можете установить их с помощью команды:

```bash
pip install -r requirements.txt
```

### Загрузка данных

Скачайте датасет FUNSD

```bash
python main/data_load.py
```

### Обучение модели

Для обучения модели выполните следующую команду:

```bash
python main/train_model.py
```

Этот скрипт выполнит следующие шаги:
- Загрузит датасет и выполнит предобработку данных.
- Определит модель LayoutLMv3 и задаст параметры обучения.
- Запустит процесс обучения и сохранит обученную модель и метрики.

### Инференс с обученной моделью

После завершения обучения модели, вы можете использовать её для выполнения инференса:

```bash
python main/predict_model.py
```

Этот скрипт выполнит следующие шаги:
- Загрузит обученную модель и процессор.
- Загрузит изображение и выполнит его предобработку.
- Выполнит предсказание и визуализирует результаты.

## Автор
Пешков Матвей (https://github.com/Peshkov-Matvei).
