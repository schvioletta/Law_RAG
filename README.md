# Law RAG — поиск судебных актов (Recall@5)

## Что сдавать

Корневой **`submission.csv` = top5f** (цель ~0.74)  
md5: `8877dbf27b9fbe58f9b3b6db0e754ace`

Откат на best LB **0.70857**: `outputs/submission_top5d_lb070857.csv`  
(md5 `352ece678faee8705ca3b9fdfe751d10`)

Безопасный fuse (overlap≈0.96): `outputs/submission_top5f_fuse.csv`

## История LB

| Версия | Идея | LB |
|--------|------|-----|
| top5 | CE@384/top60 | 0.68571 |
| top5c | BM25-doc + collapse | 0.59429 |
| **top5d** | CE@512/top70 | **0.70857** |
| top5e | retrieve150 + CE top85 | ? |
| **top5f** | neighbor ideal_answer expansion + CE@512/top90 + 2nd-chunk | **к сдаче → 0.74** |

## top5f

1. Dedup + drop train golds  
2. Neighbor expansion: top-3 похожих train-вопросов → термины из `ideal_answer`/`evidence` (без train-doc)  
3. RRF: BM25(q) + BM25(q_exp) + dense(q) + dense(q+pseudo_answer)  
4. CE `bge-reranker-v2-m3` @512 на top-90 чанков  
5. Скор документа = max(CE) + 0.15 * second_best chunk  

```bash
python3 src/run_top5f.py --ce-max-cands 90 --submit raw
```
