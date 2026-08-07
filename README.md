# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно

Файл для лидерборда: **`submission.csv`** (`qid,doc_id`, по 5 документов на вопрос).

### История метрик

| Версия | Идея | LB / локально |
|--------|------|----------------|
| train-bag LTR | признаки «документ встречался в train» | LB ≈ **0.006** |
| pure LTR (`make_submission.py`) | BM25+TFIDF+e5 без утечки | LB **0.440** |
| v5/v6 | +BGE, FT e5; CV по `gold_doc` | honest CV ≈ **0.58**, LB **0.52** |
| CE FT (v8/v9) | mmarco MiniLM на evidence | holdout **0.19** (хуже FS) — отброшен |
| HyDE ruT5 | q→ideal_answer | q+gen хуже BM25(q) |
| **v10** | FT e5 на `gold_evidence`, all-chunk dense+BM25+LTR | CV ≈0.65 (завышен), **LB 0.52** |
| **v12** | e5-base + ICT на всех 468 docs + masked evidence; pure dense | train dense@5 ≈ **0.66**, train_gold_frac≈0.53 |

Oracle: `BM25(q+ideal_answer)` ≈ **0.77**; evidence keywords ≈ **0.90**. Лидер LB **0.76**.

v10 CV был завышен: e5 видел все gold evidence до GroupKFold.

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Подход (v10)

1. Fine-tune `multilingual-e5-small` на парах (question, `gold_evidence_text`) — MNRL.
2. First-stage: BM25 doc+chunk + TF-IDF + **dense по всем чанкам** (без subsample).
3. Rerank: LightGBM LambdaRank (без train-gold bag).
4. Honest CV: **GroupKFold по `gold_doc_id`**.

Нельзя опираться на «этот doc был gold в train» — на LB это обваливает скор.

## Запуск

```bash
pip install -r requirements.txt
python3 src/make_submission_v10.py   # лучший текущий стек
```

Результат: `submission.csv`.
