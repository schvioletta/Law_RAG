"""Fine-tune e5-small on question↔gold_evidence with hard BM25 negatives.

All-chunk dense retrieval + BM25 RRF → submission. Honest GroupKFold by gold_doc.
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
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import GroupKFold
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
BASE = "intfloat/multilingual-e5-small"
SEED = 42
TOP_K = 5
CHUNK_SIZE, CHUNK_OV = 800, 120


def tok(text: str) -> list[str]:
    return [t for t in tokenize_lemmas(text) if t not in LEGAL_STOP]


def rrf(lists, k=60, weights=None):
    weights = weights or [1.0] * len(lists)
    scores = defaultdict(float)
    for w, lst in zip(weights, lists):
        for r, d in enumerate(lst):
            scores[d] += w / (k + r + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


def build_chunks(docs: pd.DataFrame):
    chunk_texts, chunk_docs = [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OV):
            chunk_texts.append(ch)
            chunk_docs.append(did)
    return chunk_texts, chunk_docs


def encode_passages(model, texts, batch_size=32):
    return model.encode(
        [f"passage: {t}" for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def encode_queries(model, questions, batch_size=32):
    return model.encode(
        [f"query: {q}" for q in questions],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def dense_doc_scores(qe, chunk_emb, chunk_docs, full_emb, doc_ids):
    sims = chunk_emb @ qe
    best = {}
    for s, d in zip(sims, chunk_docs):
        if d not in best or s > best[d]:
            best[d] = float(s)
    for i, s in enumerate(full_emb @ qe):
        best[doc_ids[i]] = max(best.get(doc_ids[i], -1e9), float(s))
    return best


def make_train_examples(train_df, bm25, doc_ids, dmap, negatives=3):
    """MNRL-style positives; hard negs via separate triplet-ish InputExamples for contrastive via MultipleNegatives — we only pass positives for MNRL (in-batch negs). Extra hard-neg pairs as (q, hard) with different queries in batch still work as in-batch."""
    examples = []
    for _, row in train_df.iterrows():
        q = str(row.question)
        ev = str(row.gold_evidence_text)
        # primary: evidence
        examples.append(InputExample(texts=[f"query: {q}", f"passage: {ev[:1800]}"]))
        # secondary: ideal answer as pseudo-passage (helps query→legal-answer space)
        examples.append(
            InputExample(
                texts=[f"query: {q}", f"passage: {str(row.ideal_answer)}"]
            )
        )
        # also a short window around evidence in the doc
        doc = dmap[row.gold_doc_id]
        idx = doc.find(ev[:80]) if len(ev) >= 80 else -1
        if idx >= 0:
            window = doc[max(0, idx - 100) : idx + min(len(ev), 1000) + 100]
            examples.append(InputExample(texts=[f"query: {q}", f"passage: {window}"]))
    return examples


def evaluate_model(model, train, docs, chunk_texts, chunk_docs, bm25, doc_ids, te_idx):
    # encode only needed? encode all docs once
    chunk_emb = encode_passages(model, chunk_texts, batch_size=64)
    full_emb = encode_passages(model, docs.text.astype(str).tolist(), batch_size=32)
    qs = train.iloc[te_idx].question.astype(str).tolist()
    qe = encode_queries(model, qs, batch_size=64)
    golds = train.iloc[te_idx].gold_doc_id.tolist()

    hit_d = hit_f = hit_b = 0
    for i, (q, g) in enumerate(zip(qs, golds)):
        dens = dense_doc_scores(qe[i], chunk_emb, chunk_docs, full_emb, doc_ids)
        dens_rank = [d for d, _ in sorted(dens.items(), key=lambda x: -x[1])]
        bm_rank = [doc_ids[j] for j in np.argsort(bm25.get_scores(tok(q)))[::-1]]
        fused = rrf([dens_rank[:200], bm_rank[:200]], weights=[1.5, 1.2])
        hit_d += g in dens_rank[:5]
        hit_b += g in bm_rank[:5]
        hit_f += g in fused[:5]
    n = len(te_idx)
    return hit_d / n, hit_b / n, hit_f / n


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    dmap = dict(zip(docs.doc_id, docs.text.astype(str)))

    print("BM25...")
    doc_toks = [tok(t) for t in tqdm(docs.text)]
    bm25 = BM25Okapi(doc_toks)
    chunk_texts, chunk_docs = build_chunks(docs)
    print(f"chunks={len(chunk_texts)}")

    # 3-fold honest CV
    gkf = GroupKFold(3)
    qidx = np.arange(len(train))
    fold_scores = []
    for fold, (tr, te) in enumerate(gkf.split(qidx, groups=train.gold_doc_id.values)):
        print(f"\n=== fold {fold} ===")
        model = SentenceTransformer(BASE)
        examples = make_train_examples(train.iloc[tr], bm25, doc_ids, dmap)
        random.shuffle(examples)
        loader = DataLoader(examples, shuffle=True, batch_size=16)
        loss = losses.MultipleNegativesRankingLoss(model)
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=3,
            warmup_steps=max(10, len(loader) // 5),
            output_path=str(OUT / f"e5_evid_fold{fold}"),
            show_progress_bar=True,
            use_amp=False,
            optimizer_params={"lr": 2e-5},
        )
        model = SentenceTransformer(str(OUT / f"e5_evid_fold{fold}"))
        d, b, f = evaluate_model(
            model, train, docs, chunk_texts, chunk_docs, bm25, doc_ids, te
        )
        print(f"fold {fold}: dense@5={d:.3f} bm25@5={b:.3f} fuse@5={f:.3f}")
        fold_scores.append(f)

    cv = float(np.mean(fold_scores))
    print(f"CV fuse R@5={cv:.4f}")

    # Final model
    print("\n=== final FT ===")
    model = SentenceTransformer(BASE)
    examples = make_train_examples(train, bm25, doc_ids, dmap)
    random.shuffle(examples)
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=4,
        warmup_steps=max(10, len(loader) // 5),
        output_path=str(OUT / "e5_evid_final"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )
    model = SentenceTransformer(str(OUT / "e5_evid_final"))

    print("encode corpus...")
    chunk_emb = encode_passages(model, chunk_texts, batch_size=64)
    full_emb = encode_passages(model, docs.text.astype(str).tolist(), batch_size=32)
    np.savez_compressed(
        CACHE / "e5_evid_emb.npz", chunk_emb=chunk_emb, full_emb=full_emb
    )

    all_q = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    all_qe = encode_queries(model, all_q, batch_size=64)
    train_qe, test_qe = all_qe[: len(train)], all_qe[len(train) :]

    # train optimistic
    tr_hit = 0
    for i, row in train.iterrows():
        dens = dense_doc_scores(train_qe[i], chunk_emb, chunk_docs, full_emb, doc_ids)
        dens_rank = [d for d, _ in sorted(dens.items(), key=lambda x: -x[1])]
        bm_rank = [
            doc_ids[j]
            for j in np.argsort(bm25.get_scores(tok(str(row.question))))[::-1]
        ]
        fused = rrf([dens_rank[:200], bm_rank[:200]], weights=[1.55, 1.15])[:TOP_K]
        tr_hit += row.gold_doc_id in fused
    print(f"train optimistic fuse@5={tr_hit/len(train):.4f}")

    rows = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        dens = dense_doc_scores(test_qe[i], chunk_emb, chunk_docs, full_emb, doc_ids)
        dens_rank = [d for d, _ in sorted(dens.items(), key=lambda x: -x[1])]
        bm_rank = [
            doc_ids[j]
            for j in np.argsort(bm25.get_scores(tok(str(row.question))))[::-1]
        ]
        fused = rrf([dens_rank[:200], bm_rank[:200]], weights=[1.55, 1.15])[:TOP_K]
        for d in fused:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    metrics = {
        "cv_fuse": cv,
        "folds": fold_scores,
        "train_optimistic": tr_hit / len(train),
        "unique_docs": int(sub.doc_id.nunique()),
        "train_gold_frac": float(sub.doc_id.isin(set(train.gold_doc_id)).mean()),
    }
    print(metrics)
    with open(OUT / "metrics_e5_evid.txt", "w") as f:
        f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
