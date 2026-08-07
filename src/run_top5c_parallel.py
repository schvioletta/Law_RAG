"""Shard top5c over CPU workers, then merge submission."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from preprocess import chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 1000
TOP_RETRIEVE = 120
RRF_K = 60
TOP_K = 5
EMBED_MODEL = "deepvk/USER2-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def rrf_fuse(lists, k=RRF_K):
    scores = defaultdict(float)
    for lst in lists:
        for rank, idx in enumerate(lst):
            scores[idx] += 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


def prepare_corpus(docs, train):
    corpus = docs.drop_duplicates(subset=["text"], keep="first").copy()
    train_docs = set(train.gold_doc_id.astype(str))
    corpus = corpus[~corpus.doc_id.astype(str).isin(train_docs)].copy()
    return corpus.reset_index(drop=True)


def build_chunks(corpus):
    texts, doc_ids = [], []
    for did, text in zip(corpus.doc_id.astype(str), corpus.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OVERLAP):
            texts.append(ch)
            doc_ids.append(did)
    return texts, doc_ids


def top_indices(scores, top_n):
    if top_n >= len(scores):
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, top_n)[:top_n]
    return idx[np.argsort(-scores[idx])].tolist()


def best_chunk_per_doc(fused, chunk_doc_ids):
    out, seen = [], set()
    for ci in fused:
        did = chunk_doc_ids[ci]
        if did in seen:
            continue
        seen.add(did)
        out.append(ci)
    return out


def worker(shard_id: int, n_shards: int, args_dict: dict):
    torch.set_num_threads(args_dict["threads_per_worker"])
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    # shard queries
    test = test.iloc[shard_id::n_shards].reset_index(drop=True)
    print(f"[shard {shard_id}] queries={len(test)}", flush=True)

    corpus = prepare_corpus(docs, train)
    chunk_texts, chunk_doc_ids = build_chunks(corpus)
    doc_ids = corpus.doc_id.astype(str).tolist()

    chunk_toks = [tokenize_lemmas(t) for t in tqdm(chunk_texts, desc=f"lem{shard_id}", leave=False)]
    doc_toks = [tokenize_lemmas(t) for t in corpus.text.astype(str)]
    bm25_chunk, bm25_doc = BM25Okapi(chunk_toks), BM25Okapi(doc_toks)

    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    emb_path = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz"
    chunk_emb = np.load(emb_path)["emb"].astype(np.float32)
    assert chunk_emb.shape[0] == len(chunk_texts)

    reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=args_dict["ce_max_length"])

    q_embs = embedder.encode(
        test.question.astype(str).tolist(),
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_query",
    )

    rows = []
    ce_max_docs = args_dict["ce_max_docs"]
    ce_max_chars = args_dict["ce_max_chars"]
    for qi, (qid, q, qe) in enumerate(
        tqdm(
            list(zip(test.qid.astype(str), test.question.astype(str), q_embs)),
            desc=f"rank{shard_id}",
        )
    ):
        q_tok = tokenize_lemmas(q)
        bm_c = top_indices(np.asarray(bm25_chunk.get_scores(q_tok), float), TOP_RETRIEVE)
        dens = top_indices(chunk_emb @ np.asarray(qe, np.float32), TOP_RETRIEVE)
        doc_rank = top_indices(np.asarray(bm25_doc.get_scores(q_tok), float), TOP_RETRIEVE)
        first = {}
        for i, did in enumerate(chunk_doc_ids):
            if did not in first:
                first[did] = i
        bm_d = [first[doc_ids[j]] for j in doc_rank if doc_ids[j] in first]
        fused = rrf_fuse([bm_c, dens, bm_d])
        cands = best_chunk_per_doc(fused, chunk_doc_ids)[:ce_max_docs]
        pairs = [(q, chunk_texts[i][:ce_max_chars]) for i in cands]
        scores = reranker.predict(
            pairs, batch_size=16, show_progress_bar=False, convert_to_numpy=True
        )
        order = np.argsort(-scores)
        top = [chunk_doc_ids[cands[i]] for i in order][:TOP_K]
        if len(top) < TOP_K:
            for did in doc_ids:
                if did not in top:
                    top.append(did)
                if len(top) >= TOP_K:
                    break
        for did in top[:TOP_K]:
            rows.append({"qid": qid, "doc_id": did})

        if (qi + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT / f"submission_top5c.shard{shard_id}.partial.csv", index=False)
            print(f"[shard {shard_id}] {qi+1}/{len(test)}", flush=True)

    out = OUT / f"submission_top5c.shard{shard_id}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[shard {shard_id}] wrote {out}", flush=True)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--ce-max-docs", type=int, default=50)
    ap.add_argument("--ce-max-chars", type=int, default=1200)
    ap.add_argument("--ce-max-length", type=int, default=512)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Ensure embeddings exist (build in parent if missing)
    emb_path = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz"
    if not emb_path.exists():
        raise SystemExit(f"missing embeddings cache {emb_path}; run make_submission_top5.py once")

    args_dict = {
        "threads_per_worker": args.threads_per_worker,
        "ce_max_docs": args.ce_max_docs,
        "ce_max_chars": args.ce_max_chars,
        "ce_max_length": args.ce_max_length,
    }

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = []
    for sid in range(args.workers):
        p = ctx.Process(target=worker, args=(sid, args.workers, args_dict))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"shard failed with code {p.exitcode}")

    # merge in original test order
    test = pd.read_csv(DATA / "test.csv")
    parts = [pd.read_csv(OUT / f"submission_top5c.shard{i}.csv") for i in range(args.workers)]
    all_rows = pd.concat(parts, ignore_index=True)
    # restore test qid order
    cat = pd.Categorical(all_rows.qid, categories=test.qid.astype(str), ordered=True)
    all_rows = all_rows.assign(_ord=cat).sort_values(["_ord"]).drop(columns="_ord")
    # within qid keep shard order (already top1..5)
    sub = all_rows.reset_index(drop=True)
    assert len(sub) == len(test) * TOP_K
    sub.to_csv(OUT / "submission_top5c.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)

    # overlap vs previous LB submit
    old = pd.read_csv(OUT / "submission_top5.csv")
    ov = []
    for q in test.qid.astype(str):
        a = set(sub.loc[sub.qid == q, "doc_id"])
        b = set(old.loc[old.qid == q, "doc_id"])
        ov.append(len(a & b) / 5)
    meta = {
        "pipeline": "top5c parallel: +BM25doc, best-chunk/doc CE@512",
        "parent_lb": 0.68571,
        "workers": args.workers,
        "ce_max_docs": args.ce_max_docs,
        "ce_max_chars": args.ce_max_chars,
        "ce_max_length": args.ce_max_length,
        "overlap_with_top5": float(np.mean(ov)),
        "rows": int(len(sub)),
        "unique_docs": int(sub.doc_id.nunique()),
    }
    (OUT / "metrics_top5c.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
