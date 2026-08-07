# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно — что сдавать

Файл для лидерборда: **`submission.csv`** (`qid,doc_id`, по 5 документов на вопрос).

**Сейчас в репозитории лежит v6 (LB ≈ 0.52).**  
Если на лидерборде **0.49714** — это был **v12** (регрессия). Пересдайте текущий `submission.csv` из ветки/PR — это откат на v6.

Не сдавайте заново `make_submission_v10.py` / `v12` без проверки: локальный CV у них завышен из‑за утечки (FT dense на всём train до GroupKFold).

### История метрик

| Версия | Идея | LB / локально |
|--------|------|----------------|
| train-bag LTR | признаки «документ встречался в train» | LB ≈ **0.006** |
| pure LTR | BM25+TFIDF+e5 без утечки | LB **0.440** |
| **v6** | BM25+BGE+FT e5+LTR (+ soft NN) | honest CV ≈ **0.58**, **LB 0.52** ← текущий submit |
| CE FT (v8/v9) | mmarco MiniLM на evidence | holdout **0.19** — отброшен |
| HyDE ruT5 | q→ideal_answer | q+gen хуже BM25(q) |
| v10 | FT e5 на evidence + LTR | CV ≈0.65 (**leak**), LB **0.52** |
| **v12** | e5-base ICT + pure dense | **LB 0.497** — хуже, откат |

Oracle-потолки (локально): BM25(q+ideal_answer) ≈ **0.77**, BM25(q+evidence) ≈ **0.90**.

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Подход (v6 — актуальный submit)

1. First-stage: BM25 (doc+chunk) + TF-IDF + dense (BGE / FT e5, в т.ч. all-chunk index).
2. Rerank: LightGBM LambdaRank (без train-gold bag).
3. Honest CV: **GroupKFold по `gold_doc_id`**.

Нельзя опираться на «этот doc был gold в train» — на LB это обваливает скор.  
Test-вопросы почти не дублируют train (char-sim ≪ paraphrase-порога), поэтому копирование gold соседей / агрессивный NN boost раздувает CV и бьёт по LB.

## Запуск

```bash
pip install -r requirements.txt
# актуальный known-good submit уже в submission.csv (v6)
# python3 src/make_submission_v6.py   # нужен cache BGE/e5_ft
```

Эксперименты v10/v12 оставлены в `src/` для истории; **не затирайте `submission.csv` ими**, пока nested-CV не покажет реальный прирост.
