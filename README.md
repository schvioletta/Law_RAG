# Law RAG — поиск судебных актов (Recall@5)

Решение задачи retrieval по корпусу обезличенных судебных актов о спорах вокруг НПФ.

## Важно — что сдавать

Файл для лидерборда: **`submission.csv`** (`qid,doc_id`, по 5 документов на вопрос).

**Актуальный LB: 0.68571 (3 место).** Цель — догнать лидеров (~0.76).

Сдавать корневой **`submission.csv`** (top5c, md5 `58a8623e98a6046d11744ebfafa78ba4`). Предыдущий LB **0.68571** сохранён в `outputs/submission_top5.csv`.

### История метрик

| Версия | Идея | LB / локально |
|--------|------|----------------|
| train-bag LTR | признаки «документ встречался в train» | LB ≈ **0.006** |
| pure LTR | BM25+TFIDF+e5 без утечки | LB **0.440** |
| v6 | BM25+BGE+FT e5+LTR | **LB 0.52** |
| v12 | e5-base ICT + pure dense | **LB 0.497** |
| **top5** | dedup + drop train + BM25/USER2 + RRF + bge-reranker (усечённый CE) | **LB 0.68571** (3 место) |
| **top5c** | +BM25(doc) в RRF, CE@512 по лучшему чанку/doc | **к сдаче** (ожидаем >0.68571) |

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
