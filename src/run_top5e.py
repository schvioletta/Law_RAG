"""top5e: push beyond LB 0.70857 with fuller chunk-CE (same safe first-stage).

Changes vs top5d (0.70857):
- first-stage top_retrieve 150 (was 100)
- CE on top 110 fused chunks (was 70), chars 1800 (was 1200), max_length 512
- still chunk-level CE (no BM25-doc / no best-chunk collapse)
- writes raw + RRF fuse with known 0.70857 submit
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
RRF_K = 60
TOP_K = 5
EMBED_MODEL = "deepvk/USER2-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
TAG = "top5e"


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

    top_retrieve = cfg["top_retrieve"]
    rows = []
    for qi, (qid, q, qe) in enumerate(
        tqdm(list(zip(test.qid.astype(str), test.question.astype(str), q_embs)), desc=f"rank{shard_id}")
    ):
        bm = top_indices(np.asarray(bm25.get_scores(tokenize_lemmas(q)), float), top_retrieve)
        dens = top_indices(emb @ np.asarray(qe, np.float32), top_retrieve)
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
            pd.DataFrame(rows).to_csv(OUT / f"submission_{TAG}.shard{shard_id}.partial.csv", index=False)
            print(f"[shard {shard_id}] {qi+1}/{len(test)}", flush=True)

    out = OUT / f"submission_{TAG}.shard{shard_id}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[shard {shard_id}] wrote {out}", flush=True)


def fuse_submissions(lists: list[pd.DataFrame], test: pd.DataFrame, weights=None) -> pd.DataFrame:
    weights = weights or [1.0] * len(lists)
    rows = []
    for q in test.qid.astype(str):
        ranked_lists = []
        for df, w in zip(lists, weights):
            lst = df.loc[df.qid == q, "doc_id"].astype(str).tolist()
            # repeat list floor(w) times for discrete RRF weighting
            n = max(1, int(round(w)))
            for _ in range(n):
                ranked_lists.append(lst)
        fused = rrf_fuse(ranked_lists)[:TOP_K]
        for did in fused:
            rows.append({"qid": q, "doc_id": did})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads-per-worker", type=int, default=4)
    ap.add_argument("--top-retrieve", type=int, default=150)
    ap.add_argument("--ce-max-cands", type=int, default=110)
    ap.add_argument("--ce-max-chars", type=int, default=1800)
    ap.add_argument("--ce-max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--fuse-base",
        default=str(OUT / "submission_top5d_lb070857.csv"),
        help="known-good LB submit to fuse with",
    )
    ap.add_argument("--submit", choices=["raw", "fuse", "fuse_heavy", "keep"], default="keep")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    emb_path = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz"
    if not emb_path.exists():
        raise SystemExit(f"missing {emb_path}")

    cfg = {
        "threads_per_worker": args.threads_per_worker,
        "top_retrieve": args.top_retrieve,
        "ce_max_cands": args.ce_max_cands,
        "ce_max_chars": args.ce_max_chars,
        "ce_max_length": args.ce_max_length,
        "batch_size": args.batch_size,
    }

    if args.workers == 1:
        # Avoid multiprocessing spawn overhead / CPU contention for single worker.
        worker(0, 1, cfg)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(i, args.workers, cfg)) for i in range(args.workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"shard failed: {p.exitcode}")

    test = pd.read_csv(DATA / "test.csv")
    parts = [pd.read_csv(OUT / f"submission_{TAG}.shard{i}.csv") for i in range(args.workers)]
    raw = pd.concat(parts, ignore_index=True)
    cat = pd.Categorical(raw.qid, categories=test.qid.astype(str), ordered=True)
    raw = raw.assign(_o=cat).sort_values(["_o"]).drop(columns="_o").reset_index(drop=True)
    assert len(raw) == len(test) * TOP_K
    raw.to_csv(OUT / f"submission_{TAG}.csv", index=False)

    base = pd.read_csv(args.fuse_base)
    fuse = fuse_submissions([base, raw], test, weights=[1, 1])
    fuse_heavy = fuse_submissions([base, raw], test, weights=[1, 2])  # prefer new CE
    fuse.to_csv(OUT / f"submission_{TAG}_fuse.csv", index=False)
    fuse_heavy.to_csv(OUT / f"submission_{TAG}_fuse_heavy.csv", index=False)

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
            "overlap_base": float(np.mean(ov)),
        }

    meta = {
        "pipeline": "top5e: retrieve150 + chunk CE@512 top110 + fuse with 0.70857",
        "parent_lb": 0.70857,
        "cfg": cfg,
        "raw": stats(raw, "top5e"),
        "fuse": stats(fuse, "top5e_fuse"),
        "fuse_heavy": stats(fuse_heavy, "top5e_fuse_heavy"),
    }

    if args.submit == "raw":
        raw.to_csv(ROOT / "submission.csv", index=False)
        chosen = "raw"
    elif args.submit == "fuse":
        fuse.to_csv(ROOT / "submission.csv", index=False)
        chosen = "fuse"
    elif args.submit == "fuse_heavy":
        fuse_heavy.to_csv(ROOT / "submission.csv", index=False)
        chosen = "fuse_heavy"
    else:
        chosen = "keep"
    meta["submission_chosen"] = chosen
    (OUT / f"metrics_{TAG}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
