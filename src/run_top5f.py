"""top5f: push 0.70857 → ~0.74 via query expansion + fuller chunk CE.

Safe rules learned from LB:
- keep drop-train + chunk CE (no BM25-doc, no best-chunk collapse)
- new signals: neighbor ideal_answer expansion (no train-doc leak)
- denser first-stage (retrieve 150) and CE top-100 @512
- per-doc score = max chunk CE + 0.15 * 2nd-best chunk CE
- also writes fuse with known 0.70857 submit
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
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

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 1000
RRF_K = 60
TOP_K = 5
EMBED_MODEL = "deepvk/USER2-base"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
TAG = "top5f"


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


def distinctive_terms(texts: list[str], top_n: int = 24) -> list[str]:
    cnt: Counter[str] = Counter()
    for t in texts:
        for tok in tokenize_lemmas(str(t)):
            if tok in STOPWORDS or len(tok) < 4:
                continue
            # drop ultra-generic legal boilerplate
            if tok in {
                "суд",
                "истец",
                "ответчик",
                "дело",
                "решение",
                "определение",
                "заявление",
                "договор",
                "лицо",
                "право",
                "требование",
            }:
                continue
            cnt[tok] += 1
    return [w for w, _ in cnt.most_common(top_n)]


def build_neighbors(train: pd.DataFrame, embedder: SentenceTransformer, top_n: int = 3):
    cache = CACHE / "train_q_emb_user2.npz"
    q_texts = train.question.astype(str).tolist()
    if cache.exists():
        z = np.load(cache)
        if len(z["emb"]) == len(train):
            q_emb = z["emb"].astype(np.float32)
        else:
            q_emb = None
    else:
        q_emb = None
    if q_emb is None:
        q_emb = embedder.encode(
            q_texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            prompt_name="search_query",
        ).astype(np.float32)
        np.savez_compressed(cache, emb=q_emb)

    answers = (
        train.ideal_answer.fillna("").astype(str)
        + " "
        + train.gold_evidence_text.fillna("").astype(str)
    ).tolist()
    return q_emb, answers


def neighbor_expand(
    q_emb: np.ndarray,
    train_q_emb: np.ndarray,
    train_answers: list[str],
    top_n: int = 3,
) -> tuple[str, list[str]]:
    sims = train_q_emb @ q_emb
    idxs = top_indices(sims, top_n)
    texts = [train_answers[i] for i in idxs]
    terms = distinctive_terms(texts, top_n=28)
    # pseudo answer = concatenation of nearest ideal/evidence snippets
    pseudo = " ".join(texts)[:1200]
    return pseudo, terms


def unique_docs_from_doc_scores(doc_scores: dict[str, float], top_k: int) -> list[str]:
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--top-retrieve", type=int, default=150)
    ap.add_argument("--ce-max-cands", type=int, default=100)
    ap.add_argument("--ce-max-chars", type=int, default=1400)
    ap.add_argument("--ce-max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--neighbors", type=int, default=3)
    ap.add_argument("--second-chunk-weight", type=float, default=0.15)
    ap.add_argument(
        "--fuse-base",
        default=str(OUT / "submission_top5d_lb070857.csv"),
    )
    ap.add_argument("--submit", choices=["raw", "fuse", "fuse_heavy", "keep"], default="keep")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--fast-only", action="store_true", help="skip CE; only expansion first-stage fuse")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    corpus = prepare_corpus(docs, train)
    chunk_texts, chunk_doc_ids = build_chunks(corpus)
    doc_ids = corpus.doc_id.astype(str).tolist()
    print(f"corpus={len(corpus)} chunks={len(chunk_texts)}")

    print("BM25...")
    chunk_toks = [tokenize_lemmas(t) for t in tqdm(chunk_texts, desc="lem")]
    bm25 = BM25Okapi(chunk_toks)

    print("USER2...")
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    emb_path = CACHE / f"user2_base_chunks_{CHUNK_SIZE}_{CHUNK_OVERLAP}_notrain.npz"
    chunk_emb = np.load(emb_path)["emb"].astype(np.float32)
    assert chunk_emb.shape[0] == len(chunk_texts)

    train_q_emb, train_answers = build_neighbors(train, embedder, top_n=args.neighbors)

    print("encode test queries...")
    test_q_embs = embedder.encode(
        test.question.astype(str).tolist(),
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_query",
    ).astype(np.float32)

    # Precompute expansions + expanded dense queries
    expansions = []
    exp_dense_texts = []
    for q, qe in zip(test.question.astype(str), test_q_embs):
        pseudo, terms = neighbor_expand(qe, train_q_emb, train_answers, top_n=args.neighbors)
        expansions.append((pseudo, terms))
        exp_dense_texts.append((q + " " + pseudo)[:1500])

    print("encode expanded dense queries...")
    exp_q_embs = embedder.encode(
        exp_dense_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt_name="search_query",
    ).astype(np.float32)

    def first_stage_lists(qi: int, q: str):
        qe = test_q_embs[qi]
        eqe = exp_q_embs[qi]
        _, terms = expansions[qi]
        q_toks = tokenize_lemmas(q)
        # weighted expansion: original tokens + distinctive neighbor terms
        exp_toks = q_toks + terms + terms  # double-weight expansion terms
        bm = top_indices(np.asarray(bm25.get_scores(q_toks), float), args.top_retrieve)
        bm_exp = top_indices(np.asarray(bm25.get_scores(exp_toks), float), args.top_retrieve)
        dens = top_indices(chunk_emb @ qe, args.top_retrieve)
        dens_exp = top_indices(chunk_emb @ eqe, args.top_retrieve)
        return bm, bm_exp, dens, dens_exp

    # Fast ensemble path (no CE): fuse known good with expansion first-stage
    base = pd.read_csv(args.fuse_base)
    # also try top5e if present
    extra_bases = []
    p_e = OUT / "submission_top5e.csv"
    if p_e.exists():
        extra_bases.append(pd.read_csv(p_e))

    fast_rows = []
    for qi, qid in enumerate(tqdm(test.qid.astype(str), desc="fast-stage")):
        q = test.question.astype(str).iloc[qi]
        bm, bm_exp, dens, dens_exp = first_stage_lists(qi, q)
        # convert chunk ranks to doc ranks
        def chunks_to_docs(chunk_idxs):
            out, seen = [], set()
            for ci in chunk_idxs:
                d = chunk_doc_ids[ci]
                if d in seen:
                    continue
                seen.add(d)
                out.append(d)
                if len(out) >= 20:
                    break
            return out

        lists = [
            base.loc[base.qid == qid, "doc_id"].astype(str).tolist(),
            chunks_to_docs(bm_exp),
            chunks_to_docs(dens_exp),
            chunks_to_docs(rrf_fuse([bm, dens, bm_exp, dens_exp])),
        ]
        for eb in extra_bases:
            lists.append(eb.loc[eb.qid == qid, "doc_id"].astype(str).tolist())
        top = rrf_fuse(lists)[:TOP_K]
        for did in top:
            fast_rows.append({"qid": qid, "doc_id": did})
    fast = pd.DataFrame(fast_rows)
    fast.to_csv(OUT / f"submission_{TAG}_fast.csv", index=False)
    print("wrote fast ensemble", OUT / f"submission_{TAG}_fast.csv")

    if args.fast_only:
        if args.submit != "keep":
            fast.to_csv(ROOT / "submission.csv", index=False)
        meta = {"pipeline": "top5f-fast expansion fuse", "submit": args.submit}
        (OUT / f"metrics_{TAG}_fast.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return

    print("load CE...")
    reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=args.ce_max_length)

    rows = []
    for qi, (qid, q) in enumerate(
        tqdm(list(zip(test.qid.astype(str), test.question.astype(str))), desc="ce-rank")
    ):
        bm, bm_exp, dens, dens_exp = first_stage_lists(qi, q)
        fused = rrf_fuse([bm, bm_exp, dens, dens_exp])[: args.ce_max_cands]
        # Single CE pass on original question (expansion is first-stage only — dual CE is too slow on CPU)
        pairs = [(q, chunk_texts[i][: args.ce_max_chars]) for i in fused]
        scores = np.asarray(
            reranker.predict(pairs, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True),
            dtype=np.float32,
        )

        # per-doc aggregation: max + w * second
        per_doc_chunks: dict[str, list[float]] = defaultdict(list)
        for ci, sc in zip(fused, scores):
            per_doc_chunks[chunk_doc_ids[ci]].append(float(sc))
        doc_scores = {}
        for did, vals in per_doc_chunks.items():
            vals = sorted(vals, reverse=True)
            sc = vals[0]
            if len(vals) > 1:
                sc += args.second_chunk_weight * vals[1]
            doc_scores[did] = sc
        top = unique_docs_from_doc_scores(doc_scores, TOP_K)
        if len(top) < TOP_K:
            for did in doc_ids:
                if did not in top:
                    top.append(did)
                if len(top) >= TOP_K:
                    break
        for did in top[:TOP_K]:
            rows.append({"qid": qid, "doc_id": did})

        if args.checkpoint_every and (qi + 1) % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(OUT / f"submission_{TAG}.partial.csv", index=False)
            print(f"checkpoint {qi+1}/{len(test)}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / f"submission_{TAG}.csv", index=False)

    def fuse_dfs(dfs, weights=None):
        weights = weights or [1] * len(dfs)
        out = []
        for q in test.qid.astype(str):
            lists = []
            for df, w in zip(dfs, weights):
                lst = df.loc[df.qid == q, "doc_id"].astype(str).tolist()
                for _ in range(max(1, int(round(w)))):
                    lists.append(lst)
            for did in rrf_fuse(lists)[:TOP_K]:
                out.append({"qid": q, "doc_id": did})
        return pd.DataFrame(out)

    fuse = fuse_dfs([base, raw, fast], [2, 2, 1])
    fuse_heavy = fuse_dfs([base, raw], [1, 2])
    fuse.to_csv(OUT / f"submission_{TAG}_fuse.csv", index=False)
    fuse_heavy.to_csv(OUT / f"submission_{TAG}_fuse_heavy.csv", index=False)

    train_docs = set(train.gold_doc_id.astype(str))

    def stats(df, name):
        ov = [
            len(set(df.loc[df.qid == q, "doc_id"].astype(str)) & set(base.loc[base.qid == q, "doc_id"].astype(str)))
            / 5
            for q in test.qid.astype(str)
        ]
        return {
            "name": name,
            "rows": int(len(df)),
            "unique_docs": int(df.doc_id.nunique()),
            "train_leak": int(df.doc_id.astype(str).isin(train_docs).sum()),
            "overlap_070857": float(np.mean(ov)),
        }

    meta = {
        "pipeline": "top5f: neighbor-answer expansion + multi-list RRF + CE@512 + 2nd-chunk boost",
        "parent_lb": 0.70857,
        "target_lb": 0.74,
        "cfg": vars(args),
        "fast": stats(fast, "fast"),
        "raw": stats(raw, "raw"),
        "fuse": stats(fuse, "fuse"),
        "fuse_heavy": stats(fuse_heavy, "fuse_heavy"),
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
