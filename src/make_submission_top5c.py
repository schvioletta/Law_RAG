"""Quality bump over 0.68571: fuller CE + doc BM25 in RRF.

Changes vs top5 (LB 0.68571):
- RRF of BM25(chunk) + USER2 + BM25(full doc)
- Keep full fused list, collapse to best-RRF chunk per doc
- CE with max_length=512 on up to 70 docs (fuller than truncated 384/60)
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


def prepare_corpus(docs, train, drop_train=True):
    corpus = docs.drop_duplicates(subset=["text"], keep="first").copy()
    if drop_train:
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


def encode_chunks(model, texts, cache_path, batch_size=16):
    if cache_path.exists():
        emb = np.load(cache_path)["emb"]
        if emb.shape[0] == len(texts):
            print(f"loaded {cache_path} {emb.shape}")
            return emb
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_document",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, emb=emb.astype(np.float32))
    return emb.astype(np.float32)


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


def rank_query(
    query,
    bm25_chunk,
    bm25_doc,
    doc_ids,
    q_emb,
    chunk_emb,
    chunk_texts,
    chunk_doc_ids,
    reranker,
    top_retrieve=TOP_RETRIEVE,
    ce_max_docs=70,
    ce_max_chars=1800,
    batch_size=16,
    top_k=TOP_K,
):
    q_tok = tokenize_lemmas(query)
    bm_c = top_indices(np.asarray(bm25_chunk.get_scores(q_tok), float), top_retrieve)
    dens = top_indices(chunk_emb @ q_emb, top_retrieve)
    doc_rank = top_indices(np.asarray(bm25_doc.get_scores(q_tok), float), top_retrieve)
    first = {}
    for i, did in enumerate(chunk_doc_ids):
        if did not in first:
            first[did] = i
    bm_d = [first[doc_ids[j]] for j in doc_rank if doc_ids[j] in first]

    fused = rrf_fuse([bm_c, dens, bm_d])
    cands = best_chunk_per_doc(fused, chunk_doc_ids)[:ce_max_docs]
    pairs = [(query, chunk_texts[i][:ce_max_chars]) for i in cands]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    order = np.argsort(-scores)
    ranked = [chunk_doc_ids[cands[i]] for i in order]
    # unique already by construction
    return ranked[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--rerank-batch-size", type=int, default=16)
    ap.add_argument("--ce-max-docs", type=int, default=70)
    ap.add_argument("--ce-max-chars", type=int, default=1800)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    corpus = prepare_corpus(docs, train, True)
    print(f"corpus {len(corpus)}")
    chunk_texts, chunk_doc_ids = build_chunks(corpus)
    doc_ids = corpus.doc_id.astype(str).tolist()
    print(f"chunks {len(chunk_texts)}")

    print("BM25...")
    chunk_toks = [tokenize_lemmas(t) for t in tqdm(chunk_texts)]
    doc_toks = [tokenize_lemmas(t) for t in tqdm(corpus.text.astype(str))]
    bm25_chunk, bm25_doc = BM25Okapi(chunk_toks), BM25Okapi(doc_toks)

    print("USER2...")
    embedder = SentenceTransformer(EMBED_MODEL, device=args.device)
    chunk_emb = encode_chunks(
        embedder,
        chunk_texts,
        CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz",
        args.embed_batch_size,
    )
    print("CE...")
    reranker = CrossEncoder(RERANK_MODEL, device=args.device, max_length=512)

    print("queries...")
    q_embs = embedder.encode(
        test.question.astype(str).tolist(),
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_query",
    )

    rows = []
    for qi, (qid, q, qe) in enumerate(
        tqdm(list(zip(test.qid.astype(str), test.question.astype(str), q_embs)), desc="rank")
    ):
        top = rank_query(
            q,
            bm25_chunk,
            bm25_doc,
            doc_ids,
            np.asarray(qe, np.float32),
            chunk_emb,
            chunk_texts,
            chunk_doc_ids,
            reranker,
            ce_max_docs=args.ce_max_docs,
            ce_max_chars=args.ce_max_chars,
            batch_size=args.rerank_batch_size,
        )
        if len(top) < TOP_K:
            for did in doc_ids:
                if did not in top:
                    top.append(did)
                if len(top) >= TOP_K:
                    break
        for did in top[:TOP_K]:
            rows.append({"qid": qid, "doc_id": did})
        if args.checkpoint_every and (qi + 1) % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(OUT / "submission_top5c.partial.csv", index=False)
            print(f"checkpoint {qi+1}/{len(test)}")

    sub = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT / "submission_top5c.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)
    meta = {
        "pipeline": "top5c: +BM25doc RRF, best-chunk/doc CE@512, 70 docs",
        "parent_lb": 0.68571,
        "n_docs": len(corpus),
        "n_chunks": len(chunk_texts),
        "ce_max_docs": args.ce_max_docs,
        "rows": len(sub),
        "unique_docs": int(sub.doc_id.nunique()),
    }
    (OUT / "metrics_top5c.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
