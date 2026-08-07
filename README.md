# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно — что сдавать

Файл для лидерборда: **`submission.csv`** (`qid,doc_id`, по 5 документов на вопрос).

**Сейчас в репозитории — v15 equal fuse** (не откат к 0.52):

`RRF([v6, v15_ltr], weights=[1.0, 1.0])`

- отличается от v6 на **231/350** вопросов по составу top-5
- `train_gold_frac ≈ 0.591` (как у v6 ≈0.589)
- локально: nested RRF ≈ **0.55** (честный GroupKFold), LTR CV ≈ **0.62**

Якорь v6 лежит в `outputs/submission_v6.csv` (LB **0.52**). Если equal уйдёт вниз — откатитесь на него.

Не сдавайте сырой v12 / train-bag: они уже давали регрессию на LB.

### История метрик

| Версия | Идея | LB / локально |
|--------|------|----------------|
| train-bag LTR | признаки «документ встречался в train» | LB ≈ **0.006** |
| pure LTR | BM25+TFIDF+e5 без утечки | LB **0.440** |
| **v6** | BM25+BGE+FT e5+LTR (+ soft NN) | honest CV ≈ **0.58**, **LB 0.52** |
| CE FT (v8/v9) | mmarco MiniLM на evidence | holdout **0.19** — отброшен |
| HyDE ruT5 | q→ideal_answer / keywords | хуже BM25(q) на honest CV |
| v10 | FT e5 на evidence + LTR | CV ≈0.65 (**leak**), LB **0.52** |
| **v12** | e5-base ICT + pure dense | **LB 0.497** — хуже |
| v13 | e5_evid + LTR; fuse с v6 | CV ≈0.71 (**leak**); raw train_gold_frac↓ |
| v14 | USER-base + e5_evid + LTR | USER@5 ≈0.33 (слабо), не сдаём |
| **v15** | nested e5-small FT + BM25 + LTR ⊕ v6 equal | nested RRF ≈**0.55**; LTR CV ≈**0.62**; set_diff≈**231** ← текущий submit |

Oracle-потолки (локально): BM25(q+ideal_answer) ≈ **0.77**, BM25(q+evidence) ≈ **0.90**.

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Подход (v15 — актуальный submit)

1. **Nested** fine-tune `multilingual-e5-small` на question↔evidence/ideal_answer (GroupKFold по `gold_doc_id`) — честный прирост BM25→RRF ≈ 0.50→0.55.
2. Full-data FT той же схемы → dense index.
3. First-stage: BM25 (doc+chunk) + TF-IDF + dense; rerank LightGBM LambdaRank.
4. **Equal fuse с pinned v6**: `RRF([v6, v15_ltr], weights=[1.0, 1.0])` — реальные set-изменения top-5 при `train_gold_frac≈0.59`.

Альтернативы в `outputs/`:
- `submission_v6.csv` — known-good LB 0.52
- `submission_v15_raw.csv` — сырой v15 LTR
- `submission_v15_aggressive.csv` — сильнее вес v15
- `submission_v15.csv` — mild fuse (часто только reorder)

Нельзя опираться на «этот doc был gold в train» — на LB это обваливает скор.

## Запуск

```bash
pip install -r requirements.txt
# 1) nested FT + embeddings (долго на CPU)
PYTHONPATH=src python3 src/make_submission_v15_nested.py
# 2) LTR + fuse (нужны outputs/e5_nested_v15 и cache)
PYTHONPATH=src python3 src/make_submission_v15.py
```

Эксперименты v10–v14 оставлены в `src/` для истории.
