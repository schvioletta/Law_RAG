# Law RAG — поиск судебных актов (Recall@5)

## Что сдавать

Корневой **`submission.csv` = top5e**  
md5: `66ae1cb79525e7e63ffcd436748b49ed`

Бэкап best LB **0.70857**: `outputs/submission_top5d_lb070857.csv`  
(md5 `352ece678faee8705ca3b9fdfe751d10`)

## История LB

| Версия | Идея | LB |
|--------|------|-----|
| top5 | CE@384 / top60 | 0.68571 |
| top5c | BM25-doc + collapse | 0.59429 |
| **top5d** | CE@512 / top70 | **0.70857** |
| **top5e** | retrieve150 + CE@512 / top85 | **к сдаче** (цель ~0.74) |

## top5e vs top5d
- first-stage top_retrieve **150** (было 100)
- CE candidates **85** (было 70), всё ещё chunk-level `@512`
- overlap с 0.70857 ≈ **0.94** (осторожный шаг)

```bash
python3 src/run_top5e.py --workers 1 --ce-max-cands 85 --top-retrieve 150 --submit raw
```
