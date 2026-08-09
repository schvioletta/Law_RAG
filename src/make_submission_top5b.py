"""Improved top-5 pipeline toward 0.72–0.76 LB.

Same recipe as top5, with quality-focused CE and a stronger first stage:
- dedup + drop train golds
- chunks 2000/1000
- RRF of: BM25(chunk) + USER2(chunk) + BM25(full doc)
- two-stage CE with BAAI/bge-reranker-v2-m3:
  1) cheap short pass over top RRF chunks
  2) full 512-token pass over survivors (multi-chunk max per doc)
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


def rrf_fuse(lists: list[list[int | str]], k: int = RRF_K) -> list:
    scores: dict = defaultdict(float)
    for lst in lists:
        for rank, idx in enumerate(lst):
            scores[idx] += 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


def prepare_corpus(docs: pd.DataFrame, train: pd.DataFrame, drop_train: bool) -> pd.DataFrame:
    corpus = docs.drop_duplicates(subset=["text"], keep="first").copy()
    if drop_train:
        train_docs = set(train.gold_doc_id.astype(str))
        corpus = corpus[~corpus.doc_id.astype(str).isin(train_docs)].copy()
    return corpus.reset_index(drop=True)


def build_chunks(corpus: pd.DataFrame) -> tuple[list[str], list[str]]:
    texts, doc_ids = [], []
    for did, text in zip(corpus.doc_id.astype(str), corpus.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OVERLAP):
            texts.append(ch)
            doc_ids.append(did)
    return texts, doc_ids


def encode_chunks(model, texts, cache_path: Path, batch_size: int = 16) -> np.ndarray:
    if cache_path.exists():
        emb = np.load(cache_path)["emb"]
        if emb.shape[0] == len(texts):
            print(f"loaded chunk embeddings: {cache_path} {emb.shape}")
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


def top_indices(scores: np.ndarray, top_n: int) -> list[int]:
    if top_n >= len(scores):
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, top_n)[:top_n]
    return idx[np.argsort(-scores[idx])].tolist()


def retrieve_bm25(bm25: BM25Okapi, query: str, top_n: int) -> list[int]:
    return top_indices(np.asarray(bm25.get_scores(tokenize_lemmas(query)), dtype=np.float64), top_n)


def retrieve_dense(q_emb: np.ndarray, chunk_emb: np.ndarray, top_n: int) -> list[int]:
    return top_indices(chunk_emb @ q_emb, top_n)


def unique_docs(ranked_chunk_idxs: list[int], chunk_doc_ids: list[str], top_k: int) -> list[str]:
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


def ce_score_pairs(reranker: CrossEncoder, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
    if not pairs:
        return np.asarray([], dtype=np.float32)
    return np.asarray(
        reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True),
        dtype=np.float32,
    )


def rank_query(
    query: str,
    bm25_chunk: BM25Okapi,
    bm25_doc: BM25Okapi,
    doc_ids: list[str],
    q_emb: np.ndarray,
    chunk_emb: np.ndarray,
    chunk_texts: list[str],
    chunk_doc_ids: list[str],
    reranker_fast: CrossEncoder,
    reranker_full: CrossEncoder,
    top_retrieve: int = TOP_RETRIEVE,
    top_k: int = TOP_K,
    stage1_cands: int = 80,
    stage2_chunks: int = 24,
    stage1_chars: int = 400,
    stage2_chars: int = 1800,
    chunks_per_doc_stage2: int = 2,
    batch_size: int = 16,
) -> list[str]:
    bm_c = retrieve_bm25(bm25_chunk, query, top_retrieve)
    dens = retrieve_dense(q_emb, chunk_emb, top_retrieve)
    # Map full-doc BM25 ranks onto representative chunks (first chunk of each doc).
    doc_rank = retrieve_bm25(bm25_doc, query, top_retrieve)
    first_chunk = {}
    for i, did in enumerate(chunk_doc_ids):
        if did not in first_chunk:
            first_chunk[did] = i
    bm_d_chunks = [first_chunk[doc_ids[j]] for j in doc_rank if doc_ids[j] in first_chunk]

    fused = rrf_fuse([bm_c, dens, bm_d_chunks])[:stage1_cands]
    if not fused:
        return []

    # Stage 1: cheap short CE over fused chunks
    pairs1 = [(query, chunk_texts[i][:stage1_chars]) for i in fused]
    s1 = ce_score_pairs(reranker_fast, pairs1, batch_size)
    order1 = np.argsort(-s1)
    stage1_ranked = [fused[i] for i in order1]

    # Keep top chunks for stage 2, ensuring diversity across docs
    stage2, seen_docs = [], defaultdict(int)
    for ci in stage1_ranked:
        did = chunk_doc_ids[ci]
        if seen_docs[did] >= chunks_per_doc_stage2:
            continue
        stage2.append(ci)
        seen_docs[did] += 1
        if len(stage2) >= stage2_chunks:
            break

    # Stage 2: full-length CE; take max score per doc
    pairs2 = [(query, chunk_texts[i][:stage2_chars]) for i in stage2]
    s2 = ce_score_pairs(reranker_full, pairs2, batch_size)
    doc_best: dict[str, float] = {}
    doc_best_chunk: dict[str, int] = {}
    for ci, sc in zip(stage2, s2):
        did = chunk_doc_ids[ci]
        if did not in doc_best or sc > doc_best[did]:
            doc_best[did] = float(sc)
            doc_best_chunk[did] = ci

    ranked_docs = sorted(doc_best.keys(), key=lambda d: -doc_best[d])
    # pad from stage1 if needed
    for ci in stage1_ranked:
        did = chunk_doc_ids[ci]
        if did not in ranked_docs:
            ranked_docs.append(did)
        if len(ranked_docs) >= top_k:
            break
    return ranked_docs[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage1-cands", type=int, default=80)
    parser.add_argument("--stage2-chunks", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--keep-train-docs", action="store_true")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    drop_train = not args.keep_train_docs
    corpus = prepare_corpus(docs, train, drop_train=drop_train)
    print(f"corpus: {len(corpus)} docs (drop_train={drop_train})")

    chunk_texts, chunk_doc_ids = build_chunks(corpus)
    doc_ids = corpus.doc_id.astype(str).tolist()
    print(f"chunks: {len(chunk_texts)}")

    print("BM25 chunk + doc...")
    chunk_toks = [tokenize_lemmas(t) for t in tqdm(chunk_texts, desc="lem-chunk")]
    doc_toks = [tokenize_lemmas(t) for t in tqdm(corpus.text.astype(str), desc="lem-doc")]
    bm25_chunk = BM25Okapi(chunk_toks)
    bm25_doc = BM25Okapi(doc_toks)

    print(f"embedder {EMBED_MODEL} on {args.device}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=args.device)
    tag = "notrain" if drop_train else "withtrain"
    emb_cache = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_{tag}.npz"
    chunk_emb = encode_chunks(embedder, chunk_texts, emb_cache, batch_size=args.embed_batch_size)

    print(f"rerankers {RERANK_MODEL}...")
    # Two CrossEncoder instances so max_length differs cleanly.
    reranker_fast = CrossEncoder(RERANK_MODEL, device=args.device, max_length=128)
    reranker_full = CrossEncoder(RERANK_MODEL, device=args.device, max_length=512)

    print(f"encoding {len(test)} test queries...")
    test_q_embs = embedder.encode(
        test.question.astype(str).tolist(),
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_query",
    )

    rows = []
    for qi, (qid, q, qe) in enumerate(
        tqdm(list(zip(test.qid.astype(str), test.question.astype(str), test_q_embs)), desc="test-rank")
    ):
        top_docs = rank_query(
            q,
            bm25_chunk,
            bm25_doc,
            doc_ids,
            np.asarray(qe, dtype=np.float32),
            chunk_emb,
            chunk_texts,
            chunk_doc_ids,
            reranker_fast,
            reranker_full,
            stage1_cands=args.stage1_cands,
            stage2_chunks=args.stage2_chunks,
            batch_size=args.rerank_batch_size,
        )
        if len(top_docs) < TOP_K:
            for did in doc_ids:
                if did not in top_docs:
                    top_docs.append(did)
                if len(top_docs) >= TOP_K:
                    break
        for did in top_docs[:TOP_K]:
            rows.append({"qid": qid, "doc_id": did})

        if args.checkpoint_every and (qi + 1) % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(OUT / "submission_top5b.partial.csv", index=False)
            print(f"checkpoint {qi + 1}/{len(test)}")

    sub = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "submission_top5b.csv"
    sub.to_csv(out_path, index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)

    meta = {
        "pipeline": "top5b: dedup→drop_train→BM25c+USER2+BM25d→RRF→2stage-bge-reranker",
        "n_corpus_docs": int(len(corpus)),
        "n_chunks": int(len(chunk_texts)),
        "drop_train": drop_train,
        "stage1_cands": args.stage1_cands,
        "stage2_chunks": args.stage2_chunks,
        "parent_lb": 0.68571,
        "submission_rows": int(len(sub)),
        "unique_docs": int(sub.doc_id.nunique()),
    }
    (OUT / "metrics_top5b.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path} and submission.csv")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
