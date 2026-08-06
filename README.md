# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно

Файл для лидерборда: **`submission.csv`**.

Старая версия ошибочно опиралась на train-документы (Recall@5 ≈ 0.006 на LB).  
Текущая версия — **чистый поиск по текстам** без утечки train-bag.

Локальная 5-fold CV: **Recall@5 ≈ 0.68**.

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Подход (`src/make_submission.py`)

1. **First-stage**: BM25 (леммы, минус канцелярит) по документам + чанкам + TF-IDF + dense `multilingual-e5-small`, RRF.
2. **Rerank**: LightGBM LambdaRank только на retrieval-признаках (без признаков «этот doc встречался в train»).
3. Выход: top-5 `doc_id` на вопрос.

## Запуск

```bash
pip install -r requirements.txt
python3 src/make_submission.py   # CV + обучение + submission.csv
```

Результат: `submission.csv`.
