"""Fast FT cross-encoder on (question, gold_evidence) → submission.

Skips slow multi-fold during train; uses 1 holdout for estimate, then full train.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import STOPWORDS, chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "дело", "г", "года", "руб", "рубль",
    "адрес", "фио", "представитель", "заседание", "протокол", "москва",
    "мещанский", "районный", "также", "данный", "настоящий", "указанный",
}
TOP_K = 5
FIRST_N = 50
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DENSE_MODEL = "intfloat/multilingual-e5-small"
SEED = 42
MAX_LEN = 256


def tok(text: str) -> list[str]:
    return [t for t in tokenize_lemmas(text) if t not in LEGAL_STOP]


def weighted(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        out.append(t)
        if len(t) >= 6:
            out.append(t)
        if len(t) >= 9:
            out.append(t)
    return out


def rrf(lists, k=60, weights=None):
    weights = weights or [1.0] * len(lists)
    scores = defaultdict(float)
    for w, lst in zip(weights, lists):
        for r, d in enumerate(lst):
            scores[d] += w / (k + r + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    dmap = dict(zip(docs.doc_id, docs.text.astype(str)))

    print("indexes...")
    doc_toks = [tok(t) for t in tqdm(docs.text, desc="tok")]
    bm25d = BM25Okapi(doc_toks)
    chunk_toks, chunk_doc, chunk_texts = [], [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 900, 150):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
            chunk_texts.append(ch)
    bm25c = BM25Okapi(chunk_toks)
    chunk_sets = [set(t) for t in chunk_toks]
    doc_chunks = defaultdict(list)
    for i, d in enumerate(chunk_doc):
        doc_chunks[d].append(i)

    cleaned = [" ".join(t) for t in doc_toks]
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=120_000)
    Dc = tfc.fit_transform(docs.text.fillna(""))

    dense = SentenceTransformer(DENSE_MODEL)
    dens_path = CACHE / "e5_small_allchunks.npz"
    dens_docs = []
    for did, text in zip(docs.doc_id, docs.text):
        dens_docs.extend([did] * len(chunk_text(str(text), 900, 150)))
    data = np.load(dens_path)
    chunk_emb, full_emb = data["chunk_emb"], data["full_emb"]
    assert len(chunk_emb) == len(dens_docs)

    all_q = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    all_qe = dense.encode(
        [f"query: {q}" for q in all_q],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    train_qe, test_qe = all_qe[: len(train)], all_qe[len(train) :]

    def best_passage(q, doc_id, maxlen=900):
        qset = set(weighted(tok(q)))
        best_t, best_s = dmap[doc_id][:maxlen], -1.0
        for i in doc_chunks[doc_id]:
            s = float(len(qset & chunk_sets[i]))
            if s > best_s:
                best_s, best_t = s, chunk_texts[i]
        return best_t[:maxlen]

    def first_stage(q, qe, n=FIRST_N):
        qt = weighted(tok(q))
        sd = bm25d.get_scores(qt)
        sc = bm25c.get_scores(qt)
        best = {}
        for i, s in enumerate(sc):
            d = chunk_doc[i]
            if d not in best or s > best[d]:
                best[d] = float(s)
        bm_doc = [doc_ids[i] for i in np.argsort(sd)[::-1][:200]]
        bm_ch = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:200]]
        sw = cosine_similarity(tfw.transform([" ".join(tok(q))]), Dw)[0]
        sch = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf_rank = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:200]],
                [doc_ids[i] for i in np.argsort(sch)[::-1][:200]],
            ]
        )
        dens_best = {}
        for s, d in zip(chunk_emb @ qe, dens_docs):
            if d not in dens_best or s > dens_best[d]:
                dens_best[d] = float(s)
        for i, s in enumerate(full_emb @ qe):
            dens_best[doc_ids[i]] = max(dens_best.get(doc_ids[i], -1e9), float(s))
        dens_rank = [d for d, _ in sorted(dens_best.items(), key=lambda x: -x[1])[:200]]
        return rrf([bm_doc, bm_ch, dens_rank, tf_rank], weights=[1.3, 1.45, 1.35, 0.75])[:n]

    print("first-stage cache...")
    cand_map = {}
    hits = defaultdict(int)
    for i, row in tqdm(train.iterrows(), total=len(train)):
        cands = first_stage(str(row.question), train_qe[i], n=60)
        cand_map[i] = cands
        for k in (5, 20, 50):
            hits[k] += row.gold_doc_id in cands[:k]
    print("FS", {k: round(hits[k] / len(train), 4) for k in hits})

    test_cands = [
        first_stage(str(row.question), test_qe[i], n=FIRST_N)
        for i, row in tqdm(test.iterrows(), total=len(test))
    ]

    def build_pairs(idxs, n_neg=4):
        ex = []
        for i in idxs:
            row = train.iloc[int(i)]
            q = str(row.question)
            gold = row.gold_doc_id
            ev = str(row.gold_evidence_text)[:1400]
            ex.append(InputExample(texts=[q, ev], label=1.0))
            # also ideal answer as medium positive
            ex.append(InputExample(texts=[q, str(row.ideal_answer)], label=0.85))
            negs = [d for d in cand_map[int(i)] if d != gold]
            random.shuffle(negs)
            for d in negs[:n_neg]:
                ex.append(InputExample(texts=[q, best_passage(q, d)], label=0.0))
        return ex

    # Holdout by gold_doc
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, te_idx = next(gss.split(train, groups=train.gold_doc_id))
    print(f"holdout train={len(tr_idx)} te={len(te_idx)}")

    examples = build_pairs(tr_idx, n_neg=5)
    random.shuffle(examples)
    print(f"CE pairs={len(examples)}")
    ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=MAX_LEN)
    loader = DataLoader(examples, shuffle=True, batch_size=24)
    ce.fit(
        train_dataloader=loader,
        epochs=2,
        warmup_steps=max(10, len(loader) // 8),
        output_path=str(OUT / "ce_holdout"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )

    def rerank(ce_model, q, cands):
        # score both BM25-best passage AND head of doc; take max
        pairs = []
        meta = []
        for d in cands:
            p1 = best_passage(q, d)
            pairs.append([q, p1])
            meta.append(d)
        scores = np.asarray(ce_model.predict(pairs, batch_size=32, show_progress_bar=False))
        # also score gold-evidence-like: first 1200 chars often has header noise; use mid window
        pairs2 = [[q, dmap[d][200:1200] if len(dmap[d]) > 1200 else dmap[d]] for d in cands]
        scores2 = np.asarray(ce_model.predict(pairs2, batch_size=32, show_progress_bar=False))
        scores = np.maximum(scores, scores2)
        return [cands[i] for i in np.argsort(-scores)], scores

    hit5 = hit_fs = 0
    for i in te_idx:
        row = train.iloc[int(i)]
        q = str(row.question)
        cands = cand_map[int(i)][:FIRST_N]
        hit_fs += row.gold_doc_id in cands[:5]
        ranked, _ = rerank(ce, q, cands)
        final = rrf([ranked, cands], weights=[2.6, 0.5])[:TOP_K]
        hit5 += row.gold_doc_id in final
    holdout = hit5 / len(te_idx)
    print(f"holdout R@5={holdout:.4f} (fs@5={hit_fs/len(te_idx):.4f})")

    # Full train
    print("full CE train...")
    examples = build_pairs(np.arange(len(train)), n_neg=5)
    random.shuffle(examples)
    ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=MAX_LEN)
    loader = DataLoader(examples, shuffle=True, batch_size=24)
    ce.fit(
        train_dataloader=loader,
        epochs=3,
        warmup_steps=max(10, len(loader) // 8),
        output_path=str(OUT / "ce_full"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )
    ce = CrossEncoder(str(OUT / "ce_full"), max_length=MAX_LEN)

    rows = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        q = str(row.question)
        cands = test_cands[i]
        ranked, _ = rerank(ce, q, cands)
        final = rrf([ranked, cands], weights=[2.6, 0.5])[:TOP_K]
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    metrics = {
        "holdout_recall@5": holdout,
        "fs": {str(k): hits[k] / len(train) for k in hits},
        "unique_docs": int(sub.doc_id.nunique()),
        "train_gold_frac": float(sub.doc_id.isin(set(train.gold_doc_id)).mean()),
    }
    print(metrics)
    with open(OUT / "metrics_ce_v9.txt", "w") as f:
        f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
