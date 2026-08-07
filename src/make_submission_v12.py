"""v12: e5-base + ICT on ALL docs + FT on masked evidence/ideal_answer.

Addresses LB stuck at 0.52: v10 CV was inflated (e5 saw all gold evidence
before GroupKFold). Here we teach the model every document via ICT so test
docs outside the 113 train golds are retrievable.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
BASE = "intfloat/multilingual-e5-base"
TOP_K, CAND = 5, 80
SEED = 42
CHUNK_SIZE, CHUNK_OV = 750, 100


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


def mask_case_facts(text: str) -> str:
    """Strip numbers/names so FT learns legal patterns, not case trivia."""
    t = str(text)
    t = re.sub(r"\b\d+[\d\s.,]*\b", " ", t)
    t = re.sub(r"\bФИО\w*\b", " ", t, flags=re.I)
    t = re.sub(r"«[^»]{1,80}»", " ", t)
    t = re.sub(r"№\s*\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text))
    return [p.strip() for p in parts if len(p.strip()) >= 40]


def build_ict_examples(chunk_texts: list[str], n: int = 3000) -> list[InputExample]:
    ex = []
    for ch in chunk_texts:
        sents = split_sentences(ch)
        if len(sents) < 2:
            continue
        # pick a sentence as pseudo-query, chunk as passage
        qi = random.randrange(len(sents))
        q = sents[qi]
        # light shorten long queries
        if len(q) > 280:
            q = q[:280]
        ex.append(InputExample(texts=[f"query: {q}", f"passage: {ch}"]))
    random.shuffle(ex)
    return ex[:n]


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
    doc_toks = [tok(t) for t in tqdm(docs.text, desc="tok")]
    bm25d = BM25Okapi(doc_toks)
    chunk_toks, chunk_doc, chunk_texts = [], [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OV):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
            chunk_texts.append(ch)
    bm25c = BM25Okapi(chunk_toks)
    print(f"chunks={len(chunk_texts)}")

    cleaned = [" ".join(t) for t in doc_toks]
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=120_000)
    Dc = tfc.fit_transform(docs.text.fillna(""))

    print("load e5-base...")
    model = SentenceTransformer(BASE)
    model.max_seq_length = 256

    # Stage A: ICT on all docs
    ict = build_ict_examples(chunk_texts, n=3500)
    print(f"ICT examples={len(ict)}")
    loader = DataLoader(ict, shuffle=True, batch_size=8)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=1,
        warmup_steps=50,
        output_path=str(OUT / "e5_base_ict"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )
    model = SentenceTransformer(str(OUT / "e5_base_ict"))
    model.max_seq_length = 256

    # Stage B: FT on masked evidence + ideal answers
    ft_ex = []
    for _, row in train.iterrows():
        q = str(row.question)
        ev = mask_case_facts(str(row.gold_evidence_text))[:1400]
        ans = mask_case_facts(str(row.ideal_answer))
        if len(ev) > 80:
            ft_ex.append(InputExample(texts=[f"query: {q}", f"passage: {ev}"]))
        ft_ex.append(InputExample(texts=[f"query: {q}", f"passage: {ans}"]))
        # also unmasked short evidence window (keeps some signal)
        raw = str(row.gold_evidence_text)
        ft_ex.append(InputExample(texts=[f"query: {q}", f"passage: {raw[:1000]}"]))
    random.shuffle(ft_ex)
    print(f"FT examples={len(ft_ex)}")
    loader = DataLoader(ft_ex, shuffle=True, batch_size=8)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=2,
        warmup_steps=40,
        output_path=str(OUT / "e5_base_v12"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 1.5e-5},
    )
    model = SentenceTransformer(str(OUT / "e5_base_v12"))
    model.max_seq_length = 256

    print("encode corpus...")
    chunk_emb = model.encode(
        [f"passage: {c}" for c in chunk_texts],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    full_emb = model.encode(
        [f"passage: {t}" for t in docs.text.astype(str)],
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.savez_compressed(CACHE / "e5_base_v12.npz", chunk_emb=chunk_emb, full_emb=full_emb)

    all_q = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    all_qe = model.encode(
        [f"query: {q}" for q in all_q],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    train_qe, test_qe = all_qe[: len(train)], all_qe[len(train) :]

    def dense_rank(qe):
        best = {}
        for s, d in zip(chunk_emb @ qe, chunk_doc):
            if d not in best or s > best[d]:
                best[d] = float(s)
        for i, s in enumerate(full_emb @ qe):
            best[doc_ids[i]] = max(best.get(doc_ids[i], -1e9), float(s))
        ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])]
        return ranked, best

    def retrieve(q: str, qe: np.ndarray, n: int = TOP_K) -> list[str]:
        qt = weighted(tok(q))
        sd = bm25d.get_scores(qt)
        sc = bm25c.get_scores(qt)
        best_ch = {}
        for i, s in enumerate(sc):
            d = chunk_doc[i]
            if d not in best_ch or s > best_ch[d]:
                best_ch[d] = float(s)
        bm_doc = [doc_ids[i] for i in np.argsort(sd)[::-1][:220]]
        bm_ch = [d for d, _ in sorted(best_ch.items(), key=lambda x: -x[1])[:220]]
        dens, _ = dense_rank(qe)
        sw = cosine_similarity(tfw.transform([" ".join(tok(q))]), Dw)[0]
        sch = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf_rank = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:200]],
                [doc_ids[i] for i in np.argsort(sch)[::-1][:200]],
            ]
        )
        return rrf(
            [bm_doc, bm_ch, dens[:220], tf_rank],
            weights=[1.25, 1.35, 1.85, 0.75],
        )[:n]

    # metrics on train (optimistic for dense FT) + GroupKFold on retrieval only
    # NOTE: dense FT used all train — report FS but also "leave-doc-out BM25" as floor
    hits = defaultdict(int)
    dens_hits = defaultdict(int)
    for i, row in tqdm(train.iterrows(), total=len(train), desc="train FS"):
        q = str(row.question)
        g = row.gold_doc_id
        fused = retrieve(q, train_qe[i], n=50)
        dens, _ = dense_rank(train_qe[i])
        for k in (1, 5, 10, 20, 50):
            hits[k] += g in fused[:k]
            dens_hits[k] += g in dens[:k]
    print("fuse FS", {k: round(hits[k] / len(train), 4) for k in sorted(hits)})
    print("dense FS", {k: round(dens_hits[k] / len(train), 4) for k in sorted(dens_hits)})

    # Diversity check vs train gold bag
    rows = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        for d in retrieve(str(row.question), test_qe[i], n=TOP_K):
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    gold = set(train.gold_doc_id)
    metrics = {
        "fuse_fs": {str(k): hits[k] / len(train) for k in sorted(hits)},
        "dense_fs": {str(k): dens_hits[k] / len(train) for k in sorted(dens_hits)},
        "unique_docs": int(sub.doc_id.nunique()),
        "train_gold_frac": float(sub.doc_id.isin(gold).mean()),
        "non_train_gold_unique": int(sub.loc[~sub.doc_id.isin(gold), "doc_id"].nunique()),
        "model": "e5-base ICT+masked-evidence FT",
    }
    print(metrics)
    with open(OUT / "metrics_v12.txt", "w") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
