"""v16: pipeline matching LB ~0.726 recipe.

1) Dedup docs by identical text (keep first)
2) Drop documents that appear as gold in train
3) Chunk 2000 / overlap 1000
4) BM25 on lemmatized chunks (stopwords removed)
5) USER2-base dense on chunks (search_document / search_query)
6) Per query: BM25 top-100 + dense top-100
7) RRF merge
8) bge-reranker-v2-m3 on fused chunks
9) Unique doc_id by descending CE score → top-5

Biggest reported levers: CE reranker + excluding train docs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from preprocess import STOPWORDS, chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "deepvk/USER2-base"
CE_MODEL = "BAAI/bge-reranker-v2-m3"
CHUNK_SIZE, CHUNK_OVERLAP = 2000, 1000
TOP_BM25, TOP_DENSE, TOP_RRF, TOP_K = 100, 100, 100, 5


def tok(text: str) -> list[str]:
    return [t for t in tokenize_lemmas(text) if t not in STOPWORDS]


def rrf(lists, k: int = 60, weights=None):
    weights = weights or [1.0] * len(lists)
    scores = defaultdict(float)
    for w, lst in zip(weights, lists):
        for r, idx in enumerate(lst):
            scores[idx] += w / (k + r + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


def build_corpus(exclude_train: bool = True):
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    n0 = len(docs)
    docs = docs.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n1 = len(docs)
    train_docs = set(train.gold_doc_id.astype(str))
    if exclude_train:
        docs = docs[~docs.doc_id.astype(str).isin(train_docs)].reset_index(drop=True)
    n2 = len(docs)
    chunk_texts, chunk_docs = [], []
    for did, text in zip(docs.doc_id.astype(str), docs.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_texts.append(ch)
            chunk_docs.append(did)
    meta = dict(
        n_raw=n0, n_dedup=n1, n_index=n2, n_chunks=len(chunk_texts),
        exclude_train=exclude_train, train_docs=len(train_docs),
    )
    return docs, chunk_texts, chunk_docs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-train-docs", action="store_true", help="ablation: do not drop train golds")
    ap.add_argument("--max-queries", type=int, default=0, help="smoke: only first N test queries")
    ap.add_argument("--rrf-k", type=int, default=TOP_RRF)
    ap.add_argument("--ce-batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-replace", action="store_true")
    args = ap.parse_args()

    exclude_train = not args.keep_train_docs
    tag = "v16" if exclude_train else "v16_with_train"
    emb_path = CACHE / f"user2_chunks_{tag}.npz"
    cand_path = CACHE / f"rrf_cands_{tag}.json"

    print("build corpus...")
    docs, chunk_texts, chunk_docs, meta = build_corpus(exclude_train=exclude_train)
    print(meta)
    chunk_docs = list(chunk_docs)

    print("BM25 index...")
    chunk_toks = [tok(t) for t in tqdm(chunk_texts, desc="lemmatize")]
    bm25 = BM25Okapi(chunk_toks)

    if emb_path.exists():
        print("load embeddings", emb_path)
        z = np.load(emb_path)
        chunk_emb = z["chunk_emb"].astype(np.float32)
        assert len(chunk_emb) == len(chunk_texts), (len(chunk_emb), len(chunk_texts))
    else:
        print("encode chunks with", EMB_MODEL)
        model = SentenceTransformer(EMB_MODEL, device=args.device)
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
            prompt_name="search_document",
        ).astype(np.float32)
        np.savez_compressed(emb_path, chunk_emb=chunk_emb)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test = pd.read_csv(DATA / "test.csv")
    if args.max_queries:
        test = test.head(args.max_queries).copy()

    print("encode queries...")
    q_model = SentenceTransformer(EMB_MODEL, device=args.device)
    questions = test.question.astype(str).tolist()
    q_emb = q_model.encode(
        questions,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt_name="search_query",
    ).astype(np.float32)
    del q_model

    print("first-stage BM25 + dense + RRF...")
    rrf_cands = {}  # qid -> list[chunk_idx]
    for i, row in tqdm(test.iterrows(), total=len(test), desc="retrieve"):
        q = str(row.question)
        qt = tok(q)
        bm_scores = bm25.get_scores(qt)
        bm_idx = np.argsort(-bm_scores)[:TOP_BM25].tolist()
        dens = chunk_emb @ q_emb[i]
        de_idx = np.argsort(-dens)[:TOP_DENSE].tolist()
        fused = rrf([bm_idx, de_idx], k=60, weights=[1.0, 1.0])[: args.rrf_k]
        rrf_cands[str(row.qid)] = fused
    cand_path.write_text(json.dumps({k: list(map(int, v)) for k, v in rrf_cands.items()}))

    print("load cross-encoder", CE_MODEL)
    ce = CrossEncoder(CE_MODEL, device=args.device, max_length=1024)

    rows = []
    ce_cache = {}  # optional per-query debug
    for i, row in tqdm(test.iterrows(), total=len(test), desc="rerank"):
        qid = str(row.qid)
        q = str(row.question)
        idxs = rrf_cands[qid]
        pairs = [(q, chunk_texts[j]) for j in idxs]
        scores = ce.predict(pairs, batch_size=args.ce_batch, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        seen = set()
        picked = []
        for oi in order:
            did = chunk_docs[idxs[int(oi)]]
            if did in seen:
                continue
            seen.add(did)
            picked.append(did)
            if len(picked) >= TOP_K:
                break
        # pad if needed (should be rare)
        if len(picked) < TOP_K:
            for j in idxs:
                did = chunk_docs[j]
                if did not in seen:
                    seen.add(did)
                    picked.append(did)
                if len(picked) >= TOP_K:
                    break
        for d in picked[:TOP_K]:
            rows.append({"qid": qid, "doc_id": d})
        ce_cache[qid] = picked[:TOP_K]

    sub = pd.DataFrame(rows)
    out_csv = OUT / f"submission_{tag}.csv"
    sub.to_csv(out_csv, index=False)
    train = pd.read_csv(DATA / "train.csv")
    train_gold = set(train.gold_doc_id.astype(str))
    metrics = dict(
        tag=tag,
        meta=meta,
        unique_docs=int(sub.doc_id.nunique()),
        train_gold_frac=float(sub.doc_id.astype(str).isin(train_gold).mean()),
        n_queries=int(test.shape[0]),
        models=dict(emb=EMB_MODEL, ce=CE_MODEL),
        note="USER2+BM25 RRF + bge-reranker-v2-m3; train docs excluded" if exclude_train
        else "ablation with train docs kept",
    )
    print(metrics)
    (OUT / f"metrics_{tag}.txt").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    if not args.no_replace and not args.max_queries and exclude_train:
        sub.to_csv(ROOT / "submission.csv", index=False)
        sub.to_csv(OUT / "submission.csv", index=False)
        print("ROOT submission.csv <-", tag)


if __name__ == "__main__":
    main()
