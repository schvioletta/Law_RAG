# Law RAG — поиск судебных актов (Recall@5)

## Что сдавать

Корневой **`submission.csv`** = **top5d** (fuller CE@512 по чанкам).  
md5: `352ece678faee8705ca3b9fdfe751d10`

Если регресс — откат на `outputs/submission_top5.csv` (LB **0.68571**, md5 `1153caca4930485ec534f0f84fb35012`).

## История LB

| Версия | Идея | LB |
|--------|------|-----|
| v6 | BM25+BGE+e5+LTR | 0.52 |
| **top5** | dedup + drop train + BM25/USER2 + RRF + bge@384/top60 | **0.68571** (3 место) |
| top5c | +BM25(doc), CE best-chunk/doc @512 | **0.59429** ← плохо, откатили |
| **top5d** | тот же first-stage что top5, CE по чанкам @512 / top70 | **к сдаче** (цель ~0.74) |

## Пайплайн top5d

1. Dedup текстов (keep first)
2. Exclude train gold docs
3. Chunks 2000 / overlap 1000
4. BM25 (леммы) + `deepvk/USER2-base` → RRF top-100+100
5. Cross-encoder `BAAI/bge-reranker-v2-m3` на fused **чанках** (`max_length=512`, top-70)
6. Уникальные `doc_id` по CE-скору → top-5

Отличие от провального top5c: **нет** BM25-doc и **нет** collapse в один чанк/doc до CE.

## Запуск

```bash
pip install -r requirements.txt
# known-good 0.68571
python3 src/make_submission_top5.py
# fuller CE (top5d)
python3 src/run_top5d_parallel.py --workers 1 --threads-per-worker 4 \
  --ce-max-cands 70 --ce-max-length 512 --submit raw
```

Бэкапы: `outputs/submission_top5.csv`, `outputs/submission_top5d_fuse.csv` (RRF с 0.68571).
