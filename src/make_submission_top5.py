"""Top-5 LB pipeline (~0.726): BM25 + USER2-base → RRF → bge-reranker-v2-m3.

Steps:
1. Drop duplicate document texts (keep first occurrence)
2. Exclude train gold documents from the corpus (test-time)
3. Chunk docs: 2000 chars, overlap 1000
4. Lexical: lemmatize + stopwords → BM25 over chunks
5. Dense: deepvk/USER2-base chunk embeddings
6. Parallel top-100 BM25 + top-100 dense
7. Fuse with RRF
8. Rerank fused chunks with BAAI/bge-reranker-v2-m3
9. Unique doc_ids in CE score order → top-5
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
TOP_RETRIEVE = 100
RRF_K = 60
TOP_K = 5
EMBED_MODEL = "deepvk/USER2-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def rrf_fuse(lists: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for lst in lists:
        for rank, idx in enumerate(lst):
            scores[idx] += 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


def prepare_corpus(docs: pd.DataFrame, train: pd.DataFrame, drop_train: bool) -> pd.DataFrame:
    corpus = docs.drop_duplicates(subset=["text"], keep="first").copy()
    if drop_train:
        train_docs = set(train.gold_doc_id.astype(str))
        corpus = corpus[~corpus.doc_id.astype(str).isin(train_docs)].copy()
    corpus = corpus.reset_index(drop=True)
    return corpus


def build_chunks(corpus: pd.DataFrame) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    doc_ids: list[str] = []
    for did, text in zip(corpus.doc_id.astype(str), corpus.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OVERLAP):
            texts.append(ch)
            doc_ids.append(did)
    return texts, doc_ids


def encode_chunks(
    model: SentenceTransformer,
    texts: list[str],
    cache_path: Path,
    batch_size: int = 16,
) -> np.ndarray:
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


def retrieve_bm25(bm25: BM25Okapi, query: str, top_n: int) -> list[int]:
    scores = bm25.get_scores(tokenize_lemmas(query))
    if top_n >= len(scores):
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, top_n)[:top_n]
    return idx[np.argsort(-scores[idx])].tolist()


def retrieve_dense(q_emb: np.ndarray, chunk_emb: np.ndarray, top_n: int) -> list[int]:
    scores = chunk_emb @ q_emb
    if top_n >= len(scores):
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, top_n)[:top_n]
    return idx[np.argsort(-scores[idx])].tolist()


def unique_docs_from_chunks(ranked_chunk_idxs: list[int], chunk_doc_ids: list[str], top_k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ci in ranked_chunk_idxs:
        did = chunk_doc_ids[ci]
        if did in seen:
            continue
        seen.add(did)
        out.append(did)
        if len(out) >= top_k:
            break
    return out


def rank_query_rrf_only(
    query: str,
    bm25: BM25Okapi,
    q_emb: np.ndarray,
    chunk_emb: np.ndarray,
    chunk_doc_ids: list[str],
    top_retrieve: int = TOP_RETRIEVE,
    top_k: int = TOP_K,
) -> list[str]:
    bm_idx = retrieve_bm25(bm25, query, top_retrieve)
    dens_idx = retrieve_dense(q_emb, chunk_emb, top_retrieve)
    fused = rrf_fuse([bm_idx, dens_idx])
    return unique_docs_from_chunks(fused, chunk_doc_ids, top_k)

def rank_query(
    query: str,
    bm25: BM25Okapi,
    q_emb: np.ndarray,
    chunk_emb: np.ndarray,
    chunk_texts: list[str],
    chunk_doc_ids: list[str],
    reranker: CrossEncoder,
    top_retrieve: int = TOP_RETRIEVE,
    top_k: int = TOP_K,
    rerank_batch_size: int = 32,
    ce_max_chars: int = 1000,
    ce_max_cands: int = 80,
) -> list[str]:
    bm_idx = retrieve_bm25(bm25, query, top_retrieve)
    dens_idx = retrieve_dense(q_emb, chunk_emb, top_retrieve)
    fused = rrf_fuse([bm_idx, dens_idx])[:ce_max_cands]
    if not fused:
        return []

    pairs = [(query, chunk_texts[i][:ce_max_chars]) for i in fused]
    ce_scores = reranker.predict(
        pairs,
        batch_size=rerank_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    order = np.argsort(-ce_scores)
    ranked = [fused[i] for i in order]
    return unique_docs_from_chunks(ranked, chunk_doc_ids, top_k)


def recall_at_k(pred_lists: list[list[str]], golds: list[str], k: int = 5) -> float:
    hits = 0
    for preds, g in zip(pred_lists, golds):
        hits += int(g in preds[:k])
    return hits / max(1, len(golds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument(
        "--keep-train-docs",
        action="store_true",
        help="Keep train gold docs in corpus (for local CV only; hurts LB)",
    )
    parser.add_argument(
        "--eval-train",
        action="store_true",
        help="Also score Recall@5 on train (requires --keep-train-docs)",
    )
    parser.add_argument("--max-train-eval", type=int, default=100)
    parser.add_argument("--ce-max-chars", type=int, default=800)
    parser.add_argument("--ce-max-cands", type=int, default=60)
    parser.add_argument("--ce-max-length", type=int, default=384)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="BM25+USER2+RRF only (fast interim submission without CE)",
    )
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=auto)")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    drop_train = not args.keep_train_docs
    corpus = prepare_corpus(docs, train, drop_train=drop_train)
    print(
        f"corpus: {len(corpus)} docs "
        f"(dedup from {len(docs)}; drop_train={drop_train})"
    )

    chunk_texts, chunk_doc_ids = build_chunks(corpus)
    print(f"chunks: {len(chunk_texts)}")

    print("building BM25 (lemmatized chunks)...")
    chunk_toks = [tokenize_lemmas(t) for t in tqdm(chunk_texts)]
    bm25 = BM25Okapi(chunk_toks)

    print(f"loading embedder {EMBED_MODEL} on {args.device}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=args.device)
    tag = "notrain" if drop_train else "withtrain"
    emb_cache = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_{tag}.npz"
    chunk_emb = encode_chunks(
        embedder, chunk_texts, emb_cache, batch_size=args.embed_batch_size
    )

    reranker = None
    if not args.skip_rerank:
        print(f"loading reranker {RERANK_MODEL} on {args.device}...")
        reranker = CrossEncoder(RERANK_MODEL, device=args.device, max_length=args.ce_max_length)

    def rank_one(query: str, q_emb: np.ndarray) -> list[str]:
        if args.skip_rerank:
            return rank_query_rrf_only(
                query, bm25, q_emb, chunk_emb, chunk_doc_ids
            )
        assert reranker is not None
        return rank_query(
            query,
            bm25,
            q_emb,
            chunk_emb,
            chunk_texts,
            chunk_doc_ids,
            reranker,
            rerank_batch_size=args.rerank_batch_size,
            ce_max_chars=args.ce_max_chars,
            ce_max_cands=args.ce_max_cands,
        )

    if args.eval_train:
        if drop_train:
            print("WARNING: train eval with drop_train=True → Recall@5≈0 expected")
        eval_df = train.head(args.max_train_eval)
        print(f"encoding {len(eval_df)} train queries...")
        q_embs = embedder.encode(
            eval_df.question.astype(str).tolist(),
            batch_size=args.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            prompt_name="search_query",
        )
        preds = []
        for q, qe in tqdm(list(zip(eval_df.question.astype(str), q_embs)), desc="train-eval"):
            preds.append(rank_one(q, np.asarray(qe, dtype=np.float32)))
        r5 = recall_at_k(preds, eval_df.gold_doc_id.astype(str).tolist(), TOP_K)
        print(f"train Recall@{TOP_K} (n={len(eval_df)}): {r5:.4f}")
        (OUT / "metrics_top5.txt").write_text(
            f"train_recall@{TOP_K}_n{len(eval_df)}={r5:.4f} drop_train={drop_train} skip_rerank={args.skip_rerank}\n",
            encoding="utf-8",
        )

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
        tqdm(
            list(zip(test.qid.astype(str), test.question.astype(str), test_q_embs)),
            desc="test-rank",
        )
    ):
        top_docs = rank_one(q, np.asarray(qe, dtype=np.float32))
        # pad if somehow fewer than TOP_K (should not happen with 334 docs)
        if len(top_docs) < TOP_K:
            for did in corpus.doc_id.astype(str):
                if did not in top_docs:
                    top_docs.append(did)
                if len(top_docs) >= TOP_K:
                    break
        for did in top_docs[:TOP_K]:
            rows.append({"qid": qid, "doc_id": did})

        if args.checkpoint_every and (qi + 1) % args.checkpoint_every == 0:
            partial = pd.DataFrame(rows)
            partial.to_csv(OUT / "submission_top5.partial.csv", index=False)
            print(f"checkpoint {qi + 1}/{len(test)} → outputs/submission_top5.partial.csv")

    sub = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out_name = "submission_top5_rrf.csv" if args.skip_rerank else "submission_top5.csv"
    out_path = OUT / out_name
    sub.to_csv(out_path, index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)

    meta = {
        "pipeline": (
            "dedup→drop_train→chunk2000/1000→BM25+USER2→RRF"
            + ("" if args.skip_rerank else "→bge-reranker-v2-m3")
        ),
        "n_corpus_docs": int(len(corpus)),
        "n_chunks": int(len(chunk_texts)),
        "drop_train": drop_train,
        "skip_rerank": args.skip_rerank,
        "embed_model": EMBED_MODEL,
        "rerank_model": None if args.skip_rerank else RERANK_MODEL,
        "top_retrieve": TOP_RETRIEVE,
        "ce_max_chars": args.ce_max_chars,
        "ce_max_cands": args.ce_max_cands,
        "ce_max_length": args.ce_max_length,
        "device": args.device,
        "submission_rows": int(len(sub)),
        "unique_qids": int(sub.qid.nunique()),
    }
    (OUT / "metrics_top5.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path} and submission.csv ({len(sub)} rows)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
