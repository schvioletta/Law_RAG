# Law RAG — поиск судебных актов (Recall@5)

## Что сдавать

Пока идёт top5e. Безопасный бэкап LB **0.70857**:
`outputs/submission_top5d_lb070857.csv` (md5 `352ece678faee8705ca3b9fdfe751d10`).

## История LB

| Версия | Идея | LB |
|--------|------|-----|
| top5 | CE@384 / top60 chunks | **0.68571** |
| top5c | BM25-doc + best-chunk/doc | **0.59429** (плохо) |
| **top5d** | CE@512 / top70 chunks | **0.70857** ← best so far |
| top5e (WIP) | retrieve150 + CE@512 / top110 + fuse | цель **~0.74** |

## Запуск top5e

```bash
python3 src/run_top5e.py --workers 1 --threads-per-worker 4 \
  --top-retrieve 150 --ce-max-cands 110 --ce-max-length 512 --submit keep
```
