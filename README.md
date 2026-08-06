# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно

Файл для лидерборда: **`submission.csv`** (`qid,doc_id`, по 5 документов на вопрос).

### История метрик

| Версия | Идея | LB / локально |
|--------|------|----------------|
| train-bag LTR | признаки «документ встречался в train» | LB ≈ **0.006** |
| pure LTR (`make_submission.py`) | BM25+TFIDF+e5 без утечки | LB **0.440** |
| v5/v6 | +BGE, FT e5, RM3/dense-all; CV по `gold_doc` | honest CV ≈ **0.58** |

Локальный CV без группировки по `gold_doc` был завышен (~0.68): у одного документа несколько train-вопросов.

Oracle: эмбеддинг `gold_evidence` даёт Recall@5 ≈ **0.79** — цель достижима, если query-эмбеддинг приблизить к evidence.

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Подход

1. **First-stage**: BM25 (леммы) по документам/чанкам + TF-IDF + dense (BGE-m3, fine-tuned e5).
2. **Rerank**: LightGBM LambdaRank на retrieval-признаках (без bag-of-train-docs).
3. Мягкий boost только при высокой похожести вопроса на train (порог ~0.52).
4. Выход: top-5 `doc_id`.

Нельзя опираться на «этот doc был gold в train» как основной сигнал — на LB это обваливает скор.

## Запуск

```bash
pip install -r requirements.txt
python3 src/make_submission.py      # базовый pure pipeline
# или более новые:
python3 src/make_submission_v6.py
python3 src/make_submission_v7.py   # после fine-tune e5_ft2
```

Результат: `submission.csv`.
