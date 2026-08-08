# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно — что сдавать

Файл для лидерборда: **`submission.csv`** (`qid,doc_id`, по 5 документов на вопрос).

**Актуальный submit = top5, LB 0.68571 (3 место).**  
top5c на LB дал **0.59429** — откатили. Цель: **~0.74**.

Сдавать корневой **`submission.csv`** (md5 `1153caca4930485ec534f0f84fb35012`).

### История метрик

| Версия | Идея | LB |
|--------|------|-----|
| v6 | BM25+BGE+FT e5+LTR | 0.52 |
| **top5** | dedup + drop train + BM25/USER2 + RRF + bge-reranker | **0.68571** ← текущий |
| top5c | +BM25(doc), CE best-chunk/doc @512 | **0.59429** (регрессия, откат) |
| top5d (WIP) | тот же first-stage, fuller CE по чанкам @512 | в работе → цель 0.74 |

## Данные

| Файл | Описание |
|------|----------|
| `data/documents.csv` | 468 документов (`doc_id`, `text`) |
| `data/train.csv` | 700 размеченных вопросов |
| `data/test.csv` | 350 вопросов для лидерборда |
| `data/sample_submission.csv` | шаблон сабмита |

## Пайплайн (top-5 / 0.726)

1. Удаляются документы с дублирующимся текстом (остаются первые вхождения).
2. Исключаются документы из тренировочной выборки (`gold_doc_id` из `train.csv`).
3. Каждый документ режется на чанки **2000** символов с перекрытием **1000**.
4. Лексический поиск: лемматизация + стоп-слова → индекс **BM25** по чанкам.
5. Семантический поиск: эмбеддинги чанков моделью **`deepvk/USER2-base`**.
6. По запросу параллельно: BM25 top-100 и dense top-100.
7. Объединение списков через **RRF**.
8. Рерanking кросс-энкодером **`BAAI/bge-reranker-v2-m3`**.
9. Сортировка по CE-скору → уникальные `doc_id` без повторений → top-5.

Наибольшее влияние на скор: **реранкер** и **удаление документов из train**.

Локальный CV на train с `drop_train=True` бессмысленен (gold вне корпуса). Для smoke-проверки используйте `--keep-train-docs --eval-train`.

## Запуск

```bash
pip install -r requirements.txt
python3 src/make_submission_top5.py
# → submission.csv и outputs/submission_top5.csv
```

Опции:

```bash
python3 src/make_submission_top5.py --device cpu --rerank-batch-size 32
python3 src/make_submission_top5.py --keep-train-docs --eval-train --max-train-eval 50
```
