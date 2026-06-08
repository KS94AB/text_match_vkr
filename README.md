# Text Match Research Service

Учебный веб-сервис для сравнительного анализа методов поиска совпадений в текстах.

## Что умеет проект
- принимает коллекцию документов `.docx` и `.txt` через веб-интерфейс;
- позволяет выбирать метод сравнения кнопкой без ручного ввода JSON;
- поддерживает основные методы `suffix_exact`, `minhash_lsh`, `inverted_index`, `ngram_jaccard`;
- сохраняет JSON API для экспериментов и автоматических тестов;
- показывает таблицу попарных сравнений и дополнительные результаты поиска по запросу.

## Быстрый запуск
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

После запуска открой:
- пользовательский интерфейс: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`

## Основные маршруты API
- `GET /health`
- `GET /methods`
- `GET /method-details`
- `POST /analyze` — JSON API
- `POST /analyze-upload` — анализ загруженной коллекции файлов
