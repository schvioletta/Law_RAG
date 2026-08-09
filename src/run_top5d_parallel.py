"""top5d: same winning first-stage as LB 0.68571, fuller chunk-level CE@512.

Does NOT use BM25-doc or best-chunk/doc collapse (those caused 0.594 regression).

Also writes a safe RRF fusion with the known-good top5 submit.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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
TOP_RETRIEVE = 100  # same as winning top5
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
    return corpus[~corpus.doc_id.astype(str).isin(train_docs)].reset_index(drop=True)


def build_chunks(corpus):
    texts, ids = [], []
    for did, text in zip(corpus.doc_id.astype(str), corpus.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OVERLAP):
            texts.append(ch)
            ids.append(did)
    return texts, ids


def top_indices(scores, top_n):
    if top_n >= len(scores):
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, top_n)[:top_n]
    return idx[np.argsort(-scores[idx])].tolist()


def unique_docs(ranked_chunk_idxs, chunk_doc_ids, top_k):
    out, seen = [], set()
    for ci in ranked_chunk_idxs:
        did = chunk_doc_ids[ci]
        if did in seen:
            continue
        seen.add(did)
        out.append(did)
        if len(out) >= top_k:
            break
    return out


def worker(shard_id: int, n_shards: int, cfg: dict):
    torch.set_num_threads(cfg["threads_per_worker"])
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv").iloc[shard_id::n_shards].reset_index(drop=True)
    print(f"[shard {shard_id}] n={len(test)}", flush=True)

    corpus = prepare_corpus(docs, train)
    chunk_texts, chunk_doc_ids = build_chunks(corpus)
    doc_ids = corpus.doc_id.astype(str).tolist()

    chunk_toks = [tokenize_lemmas(t) for t in tqdm(chunk_texts, desc=f"lem{shard_id}", leave=False)]
    bm25 = BM25Okapi(chunk_toks)

    emb = np.load(CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz")["emb"].astype(
        np.float32
    )
    assert emb.shape[0] == len(chunk_texts)

    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=cfg["ce_max_length"])

    q_embs = embedder.encode(
        test.question.astype(str).tolist(),
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_query",
    )

    rows = []
    for qi, (qid, q, qe) in enumerate(
        tqdm(list(zip(test.qid.astype(str), test.question.astype(str), q_embs)), desc=f"rank{shard_id}")
    ):
        bm = top_indices(np.asarray(bm25.get_scores(tokenize_lemmas(q)), float), TOP_RETRIEVE)
        dens = top_indices(emb @ np.asarray(qe, np.float32), TOP_RETRIEVE)
        fused = rrf_fuse([bm, dens])[: cfg["ce_max_cands"]]
        pairs = [(q, chunk_texts[i][: cfg["ce_max_chars"]]) for i in fused]
        scores = reranker.predict(
            pairs, batch_size=cfg["batch_size"], show_progress_bar=False, convert_to_numpy=True
        )
        ranked = [fused[i] for i in np.argsort(-scores)]
        top = unique_docs(ranked, chunk_doc_ids, TOP_K)
        if len(top) < TOP_K:
            for did in doc_ids:
                if did not in top:
                    top.append(did)
                if len(top) >= TOP_K:
                    break
        for did in top[:TOP_K]:
            rows.append({"qid": qid, "doc_id": did})
        if (qi + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(OUT / f"submission_top5d.shard{shard_id}.partial.csv", index=False)
            print(f"[shard {shard_id}] {qi+1}/{len(test)}", flush=True)

    out = OUT / f"submission_top5d.shard{shard_id}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[shard {shard_id}] wrote {out}", flush=True)


def fuse_submissions(base: pd.DataFrame, other: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in test.qid.astype(str):
        a = base.loc[base.qid == q, "doc_id"].astype(str).tolist()
        b = other.loc[other.qid == q, "doc_id"].astype(str).tolist()
        fused = rrf_fuse([a, b])[:TOP_K]
        for did in fused:
            rows.append({"qid": q, "doc_id": did})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--ce-max-cands", type=int, default=100)
    ap.add_argument("--ce-max-chars", type=int, default=1600)
    ap.add_argument("--ce-max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--submit",
        choices=["fuse", "raw", "keep"],
        default="fuse",
        help="fuse=RRF(top5,top5d) → submission.csv; raw=top5d; keep=do not overwrite root",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    emb_path = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz"
    if not emb_path.exists():
        raise SystemExit(f"missing {emb_path}")

    cfg = {
        "threads_per_worker": args.threads_per_worker,
        "ce_max_cands": args.ce_max_cands,
        "ce_max_chars": args.ce_max_chars,
        "ce_max_length": args.ce_max_length,
        "batch_size": args.batch_size,
    }

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i, args.workers, cfg)) for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"shard failed: {p.exitcode}")

    test = pd.read_csv(DATA / "test.csv")
    parts = [pd.read_csv(OUT / f"submission_top5d.shard{i}.csv") for i in range(args.workers)]
    raw = pd.concat(parts, ignore_index=True)
    cat = pd.Categorical(raw.qid, categories=test.qid.astype(str), ordered=True)
    raw = raw.assign(_o=cat).sort_values(["_o"]).drop(columns="_o").reset_index(drop=True)
    assert len(raw) == len(test) * TOP_K
    raw.to_csv(OUT / "submission_top5d.csv", index=False)

    base = pd.read_csv(OUT / "submission_top5.csv")  # known LB 0.68571
    fused = fuse_submissions(base, raw, test)
    fused.to_csv(OUT / "submission_top5d_fuse.csv", index=False)

    train = pd.read_csv(DATA / "train.csv")
    train_docs = set(train.gold_doc_id.astype(str))

    def stats(df, name):
        ov = []
        for q in test.qid.astype(str):
            a = set(df.loc[df.qid == q, "doc_id"].astype(str))
            b = set(base.loc[base.qid == q, "doc_id"].astype(str))
            ov.append(len(a & b) / 5)
        return {
            "name": name,
            "rows": int(len(df)),
            "unique_docs": int(df.doc_id.nunique()),
            "train_leak": int(df.doc_id.astype(str).isin(train_docs).sum()),
            "overlap_top5": float(np.mean(ov)),
        }

    meta = {
        "pipeline": "top5d: identical first-stage to 0.68571 + fuller chunk CE@512",
        "parent_lb": 0.68571,
        "failed_top5c_lb": 0.59429,
        "cfg": cfg,
        "raw": stats(raw, "top5d"),
        "fuse": stats(fused, "top5d_fuse"),
    }
    (OUT / "metrics_top5d.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.submit == "fuse":
        fused.to_csv(ROOT / "submission.csv", index=False)
        chosen = "fuse"
    elif args.submit == "raw":
        raw.to_csv(ROOT / "submission.csv", index=False)
        chosen = "raw"
    else:
        chosen = "keep"
    meta["submission_chosen"] = chosen
    (OUT / "metrics_top5d.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
