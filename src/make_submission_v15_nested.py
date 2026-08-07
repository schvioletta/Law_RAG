"""Nested e5-small FT on evidence (honest GroupKFold) + BM25 RRF.

Reports true lift vs BM25 without full-data FT leakage.
If mean CV beats ~0.55, trains on all data and writes submission_v15.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess import STOPWORDS, chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
BASE = "intfloat/multilingual-e5-small"
SEED = 42
LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "дело", "г", "года", "руб", "рубль",
    "адрес", "фио", "представитель", "заседание", "протокол", "москва",
    "мещанский", "районный", "также", "данный", "настоящий", "указанный",
}


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
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    dmap = dict(zip(docs.doc_id, docs.text.astype(str)))

    chunk_texts, chunk_docs = [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 800, 120):
            chunk_texts.append(ch)
            chunk_docs.append(did)
    print("chunks", len(chunk_texts))

    doc_toks = [tok(t) for t in docs.text]
    bm = BM25Okapi(doc_toks)
    ct, cd = [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 700, 140):
            ct.append(tok(ch))
            cd.append(did)
    bmc = BM25Okapi(ct)

    def bm25_rank(q, n=80):
        qt = weighted(tok(q))
        sd = bm.get_scores(qt)
        sc = bmc.get_scores(qt)
        best = {}
        for i, s in enumerate(sc):
            d = cd[i]
            if d not in best or s > best[d]:
                best[d] = float(s)
        return rrf(
            [
                [doc_ids[i] for i in np.argsort(sd)[::-1][:220]],
                [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:220]],
            ],
            weights=[1.0, 1.3],
        )[:n]

    def dense_rank(qe, chunk_emb, full_emb, n=80):
        best = {}
        for s, d in zip(chunk_emb @ qe, chunk_docs):
            if d not in best or s > best[d]:
                best[d] = float(s)
        for i, s in enumerate(full_emb @ qe):
            best[doc_ids[i]] = max(best.get(doc_ids[i], -1e9), float(s))
        return [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:n]]

    def make_examples(idxs):
        ex = []
        for i in idxs:
            r = train.iloc[i]
            q = str(r.question)
            ev = str(r.gold_evidence_text)[:1400]
            ans = str(r.ideal_answer)
            ex.append(InputExample(texts=[f"query: {q}", f"passage: {ev}"]))
            ex.append(InputExample(texts=[f"query: {q}", f"passage: {ans}"]))
        return ex

    fold_scores = []
    gkf = GroupKFold(5)
    for fold, (tr, te) in enumerate(gkf.split(train, groups=train.gold_doc_id.values)):
        print(f"\n=== nested fold {fold} ===")
        model = SentenceTransformer(BASE, device="cpu")
        ex = make_examples(tr)
        random.shuffle(ex)
        loader = DataLoader(ex, shuffle=True, batch_size=16)
        loss = losses.MultipleNegativesRankingLoss(model)
        # Keep training short on CPU
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=2,
            warmup_steps=20,
            show_progress_bar=True,
            use_amp=False,
        )
        print("encode corpus...")
        chunk_emb = model.encode(
            [f"passage: {t}" for t in chunk_texts],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        full_emb = model.encode(
            [f"passage: {str(t)[:3500]}" for t in docs.text],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        qemb = model.encode(
            [f"query: {q}" for q in train.iloc[te].question.astype(str)],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        hit_b = hit_d = hit_r = 0
        for j, i in enumerate(te):
            q = str(train.iloc[i].question)
            g = train.iloc[i].gold_doc_id
            bm_r = bm25_rank(q)
            dn = dense_rank(qemb[j], chunk_emb, full_emb)
            fused = rrf([bm_r, dn], weights=[1.15, 1.45])
            hit_b += g in bm_r[:5]
            hit_d += g in dn[:5]
            hit_r += g in fused[:5]
        n = len(te)
        print(f"fold{fold}: bm25={hit_b/n:.4f} dense={hit_d/n:.4f} rrf={hit_r/n:.4f}")
        fold_scores.append(dict(bm25=hit_b / n, dense=hit_d / n, rrf=hit_r / n))
        del model
        # Stop after 2 folds for speed; continue if first looks strong
        if fold >= 1 and np.mean([s["rrf"] for s in fold_scores]) < 0.52:
            print("weak after 2 folds, continuing one more for stability")
        if fold >= 2:
            break

    mean_rrf = float(np.mean([s["rrf"] for s in fold_scores]))
    mean_bm = float(np.mean([s["bm25"] for s in fold_scores]))
    print("MEAN nested", mean_bm, mean_rrf)
    metrics = {"folds": fold_scores, "mean_bm25": mean_bm, "mean_rrf": mean_rrf}
    (OUT / "metrics_nested_e5.txt").write_text(json.dumps(metrics, indent=2))

    if mean_rrf < mean_bm + 0.02:
        print("No clear nested lift; skip full FT submit")
        return

    print("Full-data FT for submission...")
    model = SentenceTransformer(BASE, device="cpu")
    ex = make_examples(np.arange(len(train)))
    random.shuffle(ex)
    loader = DataLoader(ex, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=3,
        warmup_steps=50,
        show_progress_bar=True,
        use_amp=False,
    )
    model.save(str(OUT / "e5_nested_v15"))
    chunk_emb = model.encode(
        [f"passage: {t}" for t in chunk_texts],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    full_emb = model.encode(
        [f"passage: {str(t)[:3500]}" for t in docs.text],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.savez_compressed(CACHE / "e5_nested_v15.npz", chunk_emb=chunk_emb, full_emb=full_emb)

    allq = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    qemb = model.encode(
        [f"query: {q}" for q in allq],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    train_q, test_q = qemb[: len(train)], qemb[len(train) :]

    # Fuse with existing v6 on test
    v6 = pd.read_csv(ROOT / "submission.csv")
    v6l = {qid: g.doc_id.tolist() for qid, g in v6.groupby("qid", sort=False)}
    rows = []
    for i, row in test.iterrows():
        q = str(row.question)
        bm_r = bm25_rank(q)
        dn = dense_rank(test_q[i], chunk_emb, full_emb)
        fused = rrf([bm_r, dn, v6l[row.qid]], weights=[1.1, 1.5, 1.2])[:5]
        for d in fused:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission_v15.csv", index=False)
    print("wrote submission_v15", sub.doc_id.nunique(), float(sub.doc_id.isin(set(train.gold_doc_id)).mean()))


if __name__ == "__main__":
    main()
