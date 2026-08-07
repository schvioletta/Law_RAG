"""v8: hybrid first-stage (BM25+TFIDF+e5 all-chunks) + fine-tuned CE rerank.

No train-gold bag leakage. Honest GroupKFold by gold_doc_id.
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
FIRST_N = 50
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DENSE_MODEL = "intfloat/multilingual-e5-small"
SEED = 42
CHUNK_SIZE = 900
CHUNK_OV = 150


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


def best_chunk_for_evidence(doc_text: str, evidence: str, size=CHUNK_SIZE, overlap=CHUNK_OV) -> str:
    ev = str(evidence)
    chunks = chunk_text(doc_text, size, overlap)
    if not chunks:
        return str(doc_text)[:size]
    best, best_s = chunks[0], -1.0
    for ch in chunks:
        score = 0
        for n in (80, 120, 200):
            if len(ev) >= n and ev[:n] in ch:
                score += n
            if len(ev) >= n and ev[-n:] in ch:
                score += n
        mid = ev[max(0, len(ev) // 2 - 60) : len(ev) // 2 + 60]
        if mid and mid in ch:
            score += 100
        if score > best_s:
            best_s, best = score, ch
    if best_s <= 0:
        idx = str(doc_text).find(ev[:100]) if len(ev) >= 100 else -1
        if idx >= 0:
            a = max(0, idx - 100)
            return str(doc_text)[a : a + size]
    return best


class HybridIndex:
    def __init__(self, docs: pd.DataFrame):
        self.docs = docs
        self.doc_ids = docs.doc_id.tolist()
        self.id2i = {d: i for i, d in enumerate(self.doc_ids)}
        self.dmap = dict(zip(docs.doc_id, docs.text.astype(str)))

        print("BM25 indexes...")
        self.doc_toks = [tok(t) for t in tqdm(docs.text, desc="tok docs")]
        self.bm25_doc = BM25Okapi(self.doc_toks)

        self.chunk_toks, self.chunk_doc, self.chunk_texts = [], [], []
        for did, text in tqdm(list(zip(docs.doc_id, docs.text)), desc="tok chunks"):
            for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OV):
                self.chunk_toks.append(tok(ch))
                self.chunk_doc.append(did)
                self.chunk_texts.append(ch)
        self.bm25_chunk = BM25Okapi(self.chunk_toks)
        print(f"chunks: {len(self.chunk_texts)}")

        cleaned = [" ".join(t) for t in self.doc_toks]
        self.tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
        self.Dw = self.tfw.fit_transform(cleaned)
        self.tfc = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000
        )
        self.Dc = self.tfc.fit_transform(docs.text.fillna(""))

        print("Dense e5-small...")
        self.dense = SentenceTransformer(DENSE_MODEL)
        self.chunk_emb, self.full_emb, self.dens_docs = self._load_or_encode_dense()

        # precompute token sets per chunk for fast passage selection
        self.chunk_tok_sets = [set(t) for t in self.chunk_toks]
        self.doc_to_chunk_idxs = defaultdict(list)
        for i, d in enumerate(self.chunk_doc):
            self.doc_to_chunk_idxs[d].append(i)

    def _load_or_encode_dense(self):
        path = CACHE / "e5_small_allchunks.npz"
        dens_docs = []
        passages = []
        for did, text in zip(self.docs.doc_id, self.docs.text):
            chs = chunk_text(str(text), CHUNK_SIZE, CHUNK_OV)
            dens_docs.extend([did] * len(chs))
            passages.extend([f"passage: {c}" for c in chs])
        full_passages = [f"passage: {t}" for t in self.docs.text.astype(str)]

        if path.exists():
            data = np.load(path)
            if len(data["chunk_emb"]) == len(passages):
                print(f"loaded dense cache {path}")
                return data["chunk_emb"], data["full_emb"], dens_docs

        print(f"encoding {len(passages)} chunks + {len(full_passages)} docs...")
        chunk_emb = self.dense.encode(
            passages,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        full_emb = self.dense.encode(
            full_passages,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.savez_compressed(path, chunk_emb=chunk_emb, full_emb=full_emb)
        return chunk_emb, full_emb, dens_docs

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        return self.dense.encode(
            [f"query: {q}" for q in questions],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def first_stage_from_qe(self, q: str, qe: np.ndarray, n: int = FIRST_N) -> list[str]:
        qt = weighted(tok(q))
        sd = self.bm25_doc.get_scores(qt)
        sc = self.bm25_chunk.get_scores(qt)
        best_chunk = {}
        for i, s in enumerate(sc):
            d = self.chunk_doc[i]
            if d not in best_chunk or s > best_chunk[d]:
                best_chunk[d] = float(s)
        bm_doc = [self.doc_ids[i] for i in np.argsort(sd)[::-1][:200]]
        bm_ch = [d for d, _ in sorted(best_chunk.items(), key=lambda x: -x[1])[:200]]

        sw = cosine_similarity(self.tfw.transform([" ".join(tok(q))]), self.Dw)[0]
        sch = cosine_similarity(self.tfc.transform([q]), self.Dc)[0]
        tf_rank = rrf(
            [
                [self.doc_ids[i] for i in np.argsort(sw)[::-1][:200]],
                [self.doc_ids[i] for i in np.argsort(sch)[::-1][:200]],
            ]
        )

        sims = self.chunk_emb @ qe
        dens_best = {}
        for s, d in zip(sims, self.dens_docs):
            if d not in dens_best or s > dens_best[d]:
                dens_best[d] = float(s)
        full = self.full_emb @ qe
        for i, s in enumerate(full):
            dens_best[self.doc_ids[i]] = max(
                dens_best.get(self.doc_ids[i], -1e9), float(s)
            )
        dens_rank = [d for d, _ in sorted(dens_best.items(), key=lambda x: -x[1])[:200]]

        return rrf(
            [bm_doc, bm_ch, dens_rank, tf_rank],
            weights=[1.3, 1.4, 1.35, 0.8],
        )[:n]

    def best_passage(self, q: str, doc_id: str, maxlen: int = 1200) -> str:
        qset = set(weighted(tok(q)))
        best_t, best_s = self.dmap[doc_id][:maxlen], -1.0
        for i in self.doc_to_chunk_idxs[doc_id]:
            s = float(len(qset & self.chunk_tok_sets[i]))
            if s > best_s:
                best_s = s
                best_t = self.chunk_texts[i]
        return best_t[:maxlen]


def build_ce_pairs(
    train: pd.DataFrame,
    index: HybridIndex,
    tr_idx: np.ndarray,
    cand_map: dict[int, list[str]],
    negatives: int = 4,
) -> list[InputExample]:
    examples = []
    dmap = index.dmap
    for i in tr_idx:
        i = int(i)
        row = train.iloc[i]
        q = str(row.question)
        gold = row.gold_doc_id
        pos = best_chunk_for_evidence(dmap[gold], str(row.gold_evidence_text))
        examples.append(InputExample(texts=[q, pos], label=1.0))
        # shorter evidence window as extra positive
        ev = str(row.gold_evidence_text)
        if len(ev) > 200:
            examples.append(InputExample(texts=[q, ev[:1400]], label=1.0))

        cands = cand_map[i]
        negs = [d for d in cands if d != gold]
        random.shuffle(negs)
        for d in negs[:negatives]:
            examples.append(InputExample(texts=[q, index.best_passage(q, d)], label=0.0))
        # random hard-ish negatives from corpus
        for _ in range(1):
            d = random.choice(index.doc_ids)
            if d != gold:
                examples.append(
                    InputExample(texts=[q, index.dmap[d][:1200]], label=0.0)
                )
    return examples


def ce_rerank(ce: CrossEncoder, index: HybridIndex, q: str, cands: list[str]) -> list[str]:
    pairs = [[q, index.best_passage(q, d)] for d in cands]
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))
    return [cands[i] for i in order]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    index = HybridIndex(docs)

    all_q = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    print("encode all queries...")
    all_qe = index.encode_queries(all_q)
    train_qe = all_qe[: len(train)]
    test_qe = all_qe[len(train) :]

    print("precompute first-stage candidates...")
    cand_map: dict[int, list[str]] = {}
    hits = defaultdict(int)
    for i, row in tqdm(train.iterrows(), total=len(train), desc="first-stage train"):
        cands = index.first_stage_from_qe(str(row.question), train_qe[i], n=50)
        cand_map[i] = cands
        g = row.gold_doc_id
        for k in (1, 5, 10, 20, 50):
            hits[k] += g in cands[:k]
    print("first-stage:", {k: round(hits[k] / len(train), 4) for k in sorted(hits)})

    test_cands = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="first-stage test"):
        test_cands.append(
            index.first_stage_from_qe(str(row.question), test_qe[i], n=FIRST_N)
        )

    groups = train.gold_doc_id.values
    gkf = GroupKFold(5)
    qidx = np.arange(len(train))
    fold_scores = []

    for fold, (tr, te) in enumerate(gkf.split(qidx, groups=groups)):
        print(f"\n=== fold {fold} train_q={len(tr)} te_q={len(te)} ===")
        examples = build_ce_pairs(train, index, tr, cand_map, negatives=5)
        random.shuffle(examples)
        print(f"CE pairs: {len(examples)}")

        ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=512)
        loader = DataLoader(examples, shuffle=True, batch_size=16)
        ce.fit(
            train_dataloader=loader,
            epochs=2,
            warmup_steps=max(10, len(loader) // 10),
            output_path=str(OUT / f"ce_fold{fold}"),
            show_progress_bar=True,
            use_amp=False,
            optimizer_params={"lr": 2e-5},
        )

        hit5 = 0
        hit5_fs = 0
        for i in te:
            row = train.iloc[int(i)]
            q = str(row.question)
            cands = cand_map[int(i)][:FIRST_N]
            hit5_fs += int(row.gold_doc_id in cands[:5])
            ranked = ce_rerank(ce, index, q, cands)
            final = rrf([ranked, cands], weights=[2.4, 0.6])[:TOP_K]
            hit5 += int(row.gold_doc_id in final)
        score = hit5 / len(te)
        print(
            f"fold {fold}: honest R@5={score:.4f} first-stage@5={hit5_fs/len(te):.4f}"
        )
        fold_scores.append(score)

    cv = float(np.mean(fold_scores))
    print(f"\nCV honest Recall@5 = {cv:.4f}")

    print("\n=== final CE on all train ===")
    examples = build_ce_pairs(train, index, qidx, cand_map, negatives=5)
    random.shuffle(examples)
    ce = CrossEncoder(CE_MODEL, num_labels=1, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    ce.fit(
        train_dataloader=loader,
        epochs=2,
        warmup_steps=max(10, len(loader) // 10),
        output_path=str(OUT / "ce_final"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )
    ce = CrossEncoder(str(OUT / "ce_final"), max_length=512)

    rows = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test predict"):
        q = str(row.question)
        cands = test_cands[i]
        ranked = ce_rerank(ce, index, q, cands)
        final = rrf([ranked, cands], weights=[2.4, 0.6])[:TOP_K]
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    assert len(sub) == len(test) * TOP_K
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    frac = float(sub.doc_id.isin(set(train.gold_doc_id)).mean())
    metrics = {
        "cv_honest_recall@5": cv,
        "train_gold_frac": frac,
        "unique_docs": int(sub.doc_id.nunique()),
        "first_stage": {str(k): hits[k] / len(train) for k in sorted(hits)},
        "fold_scores": fold_scores,
    }
    print(metrics)
    with open(OUT / "metrics_v8.txt", "w") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
