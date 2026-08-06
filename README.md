# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Подход

1. **First-stage**: BM25 по полным документам + чанкам (лемматизация pymorphy3), RRF-слияние, буст от TF-IDF по «сумкам» train-вопросов и evidence на документ.
2. **Признаки**: BM25, TF-IDF (word/char), dense `multilingual-e5-small` (chunk/full/evidence), пересечение лемм, длина документа, покрытие train-разметкой.
3. **Rerank**: LightGBM LambdaRank → top-5 `doc_id`.

Метрика: **Recall@5**.

## Запуск

```bash
pip install -r requirements.txt
python src/retriever.py                 # CV + обучение + submission.csv
python src/retriever.py --skip-cv       # быстрее, без кросс-валидации
python src/retriever.py --cv-only       # только оценка
python src/retriever.py --no-dense      # без эмбеддингов
```

Результат: `submission.csv` и `outputs/submission.csv`.

## Формат сабмита

```csv
qid,doc_id
q_...,d_...
```

До 5 документов на вопрос, по убыванию релевантности.
