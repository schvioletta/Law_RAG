"""Faster CE pipeline: shorter seqs, 1 epoch, 2-fold CV, then full train → submit."""

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
TOP_K = 5
FIRST_N = 40
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DENSE_MODEL = "intfloat/multilingual-e5-small"
SEED = 42
CHUNK_SIZE, CHUNK_OV = 900, 150
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


def best_chunk_for_evidence(doc_text: str, evidence: str) -> str:
    ev = str(evidence)
    chunks = chunk_text(doc_text, CHUNK_SIZE, CHUNK_OV)
    best, best_s = chunks[0] if chunks else str(doc_text)[:CHUNK_SIZE], -1
    for ch in chunks:
        score = 0
        for n in (80, 120):
            if len(ev) >= n and (ev[:n] in ch or ev[-n:] in ch):
                score += n
        if score > best_s:
            best_s, best = score, ch
    return best[:800]


class HybridIndex:
    def __init__(self, docs: pd.DataFrame):
        self.docs = docs
        self.doc_ids = docs.doc_id.tolist()
        self.dmap = dict(zip(docs.doc_id, docs.text.astype(str)))
        print("BM25...")
        self.doc_toks = [tok(t) for t in tqdm(docs.text, desc="tok docs")]
        self.bm25_doc = BM25Okapi(self.doc_toks)
        self.chunk_toks, self.chunk_doc, self.chunk_texts = [], [], []
        for did, text in zip(docs.doc_id, docs.text):
            for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OV):
                self.chunk_toks.append(tok(ch))
                self.chunk_doc.append(did)
                self.chunk_texts.append(ch)
        self.bm25_chunk = BM25Okapi(self.chunk_toks)
        self.chunk_tok_sets = [set(t) for t in self.chunk_toks]
        self.doc_to_chunk_idxs = defaultdict(list)
        for i, d in enumerate(self.chunk_doc):
            self.doc_to_chunk_idxs[d].append(i)

        cleaned = [" ".join(t) for t in self.doc_toks]
        self.tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
        self.Dw = self.tfw.fit_transform(cleaned)
        self.tfc = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=120_000
        )
        self.Dc = self.tfc.fit_transform(docs.text.fillna(""))

        self.dense = SentenceTransformer(DENSE_MODEL)
        self.chunk_emb, self.full_emb, self.dens_docs = self._load_dense()

    def _load_dense(self):
        path = CACHE / "e5_small_allchunks.npz"
        dens_docs = []
        for did, text in zip(self.docs.doc_id, self.docs.text):
            dens_docs.extend([did] * len(chunk_text(str(text), CHUNK_SIZE, CHUNK_OV)))
        data = np.load(path)
        return data["chunk_emb"], data["full_emb"], dens_docs

    def encode_queries(self, questions):
        return self.dense.encode(
            [f"query: {q}" for q in questions],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def first_stage(self, q, qe, n=FIRST_N):
        qt = weighted(tok(q))
        sd = self.bm25_doc.get_scores(qt)
        sc = self.bm25_chunk.get_scores(qt)
        best_chunk = {}
        for i, s in enumerate(sc):
            d = self.chunk_doc[i]
            if d not in best_chunk or s > best_chunk[d]:
                best_chunk[d] = float(s)
        bm_doc = [self.doc_ids[i] for i in np.argsort(sd)[::-1][:180]]
        bm_ch = [d for d, _ in sorted(best_chunk.items(), key=lambda x: -x[1])[:180]]
        sw = cosine_similarity(self.tfw.transform([" ".join(tok(q))]), self.Dw)[0]
        sch = cosine_similarity(self.tfc.transform([q]), self.Dc)[0]
        tf_rank = rrf(
            [
                [self.doc_ids[i] for i in np.argsort(sw)[::-1][:180]],
                [self.doc_ids[i] for i in np.argsort(sch)[::-1][:180]],
            ]
        )
        dens_best = {}
        for s, d in zip(self.chunk_emb @ qe, self.dens_docs):
            if d not in dens_best or s > dens_best[d]:
                dens_best[d] = float(s)
        for i, s in enumerate(self.full_emb @ qe):
            dens_best[self.doc_ids[i]] = max(
                dens_best.get(self.doc_ids[i], -1e9), float(s)
            )
        dens_rank = [d for d, _ in sorted(dens_best.items(), key=lambda x: -x[1])[:180]]
        return rrf(
            [bm_doc, bm_ch, dens_rank, tf_rank], weights=[1.3, 1.4, 1.35, 0.8]
        )[:n]

    def best_passage(self, q, doc_id, maxlen=700):
        qset = set(weighted(tok(q)))
        best_t, best_s = self.dmap[doc_id][:maxlen], -1.0
        for i in self.doc_to_chunk_idxs[doc_id]:
            s = float(len(qset & self.chunk_tok_sets[i]))
            if s > best_s:
                best_s, best_t = s, self.chunk_texts[i]
        return best_t[:maxlen]


def build_pairs(train, index, idxs, cand_map, negatives=3):
    ex = []
    for i in idxs:
        i = int(i)
        row = train.iloc[i]
        q = str(row.question)
        gold = row.gold_doc_id
        pos = best_chunk_for_evidence(index.dmap[gold], str(row.gold_evidence_text))
        ex.append(InputExample(texts=[q, pos], label=1.0))
        ex.append(InputExample(texts=[q, str(row.gold_evidence_text)[:800]], label=1.0))
        negs = [d for d in cand_map[i] if d != gold]
        random.shuffle(negs)
        for d in negs[:negatives]:
            ex.append(InputExample(texts=[q, index.best_passage(q, d)], label=0.0))
    return ex


def ce_rerank(ce, index, q, cands):
    pairs = [[q, index.best_passage(q, d)] for d in cands]
    scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
    return [cands[i] for i in np.argsort(-np.asarray(scores))]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    index = HybridIndex(docs)

    all_q = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    all_qe = index.encode_queries(all_q)
    train_qe, test_qe = all_qe[: len(train)], all_qe[len(train) :]

    cand_map = {}
    hits = defaultdict(int)
    for i, row in tqdm(train.iterrows(), total=len(train), desc="fs train"):
        cands = index.first_stage(str(row.question), train_qe[i], n=50)
        cand_map[i] = cands
        for k in (5, 20, 50):
            hits[k] += row.gold_doc_id in cands[:k]
    print("first-stage", {k: hits[k] / len(train) for k in hits})

    test_cands = [
        index.first_stage(str(row.question), test_qe[i], n=FIRST_N)
        for i, row in tqdm(test.iterrows(), total=len(test), desc="fs test")
    ]

    # 2-fold honest CV
    gkf = GroupKFold(2)
    fold_scores = []
    qidx = np.arange(len(train))
    for fold, (tr, te) in enumerate(gkf.split(qidx, groups=train.gold_doc_id.values)):
        print(f"=== fold {fold} ===")
        examples = build_pairs(train, index, tr, cand_map)
        random.shuffle(examples)
        ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=MAX_LEN)
        loader = DataLoader(examples, shuffle=True, batch_size=32)
        ce.fit(
            train_dataloader=loader,
            epochs=1,
            warmup_steps=max(5, len(loader) // 10),
            output_path=str(OUT / f"ce_fast_fold{fold}"),
            show_progress_bar=True,
            use_amp=False,
            optimizer_params={"lr": 2e-5},
        )
        hit = 0
        for i in te:
            row = train.iloc[int(i)]
            cands = cand_map[int(i)][:FIRST_N]
            ranked = ce_rerank(ce, index, str(row.question), cands)
            final = rrf([ranked, cands], weights=[2.5, 0.5])[:TOP_K]
            hit += row.gold_doc_id in final
        score = hit / len(te)
        print(f"fold {fold} honest R@5={score:.4f}")
        fold_scores.append(score)

    cv = float(np.mean(fold_scores))
    print(f"CV={cv:.4f}")

    examples = build_pairs(train, index, qidx, cand_map)
    random.shuffle(examples)
    ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=MAX_LEN)
    loader = DataLoader(examples, shuffle=True, batch_size=32)
    ce.fit(
        train_dataloader=loader,
        epochs=2,
        warmup_steps=max(5, len(loader) // 10),
        output_path=str(OUT / "ce_fast_final"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )
    ce = CrossEncoder(str(OUT / "ce_fast_final"), max_length=MAX_LEN)

    rows = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="predict"):
        cands = test_cands[i]
        ranked = ce_rerank(ce, index, str(row.question), cands)
        final = rrf([ranked, cands], weights=[2.5, 0.5])[:TOP_K]
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission_ce.csv", index=False)
    metrics = {
        "cv_honest": cv,
        "folds": fold_scores,
        "unique_docs": int(sub.doc_id.nunique()),
        "train_gold_frac": float(sub.doc_id.isin(set(train.gold_doc_id)).mean()),
    }
    print(metrics)
    with open(OUT / "metrics_ce_fast.txt", "w") as f:
        f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
