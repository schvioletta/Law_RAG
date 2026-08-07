"""Pure document retrieval for Law RAG (no train-doc bag leakage)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from preprocess import chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

TOP_K = 5
CAND_N = 40


def rrf_fuse(rank_lists: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[str]:
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[str, float] = defaultdict(float)
    for w, lst in zip(weights, rank_lists):
        for rank, doc in enumerate(lst):
            scores[doc] += w / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]


class PureRetriever:
    def __init__(self, use_dense: bool = True, use_ce: bool = True):
        self.docs = pd.read_csv(DATA / "documents.csv")
        self.train = pd.read_csv(DATA / "train.csv")
        self.test = pd.read_csv(DATA / "test.csv")
        self.doc_ids = self.docs.doc_id.tolist()
        self.id2i = {d: i for i, d in enumerate(self.doc_ids)}
        self.dmap = dict(zip(self.docs.doc_id, self.docs.text.astype(str)))

        print("Building BM25 indexes...")
        self.doc_tokens = [
            tokenize_lemmas(t) for t in tqdm(self.docs.text, desc="tokenize docs")
        ]
        self.bm25_doc = BM25Okapi(self.doc_tokens)

        self.chunk_tokens: list[list[str]] = []
        self.chunk_doc: list[str] = []
        self.chunk_texts: list[str] = []
        for did, text in tqdm(
            list(zip(self.docs.doc_id, self.docs.text)), desc="chunk docs"
        ):
            for ch in chunk_text(str(text), 1100, 200):
                self.chunk_tokens.append(tokenize_lemmas(ch))
                self.chunk_doc.append(did)
                self.chunk_texts.append(ch)
        self.bm25_chunk = BM25Okapi(self.chunk_tokens)

        self.tf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100_000
        )
        self.D_word = self.tf_word.fit_transform(self.docs.text.fillna(""))
        self.tf_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=120_000
        )
        self.D_char = self.tf_char.fit_transform(self.docs.text.fillna(""))

        self.use_dense = use_dense
        self.use_ce = use_ce
        self.dense = None
        self.chunk_emb = None
        self.full_emb = None
        self.ce = None
        self.query_emb_cache: dict[str, np.ndarray] = {}

        if use_dense:
            self._init_dense()
        if use_ce:
            print("Loading cross-encoder...")
            self.ce = CrossEncoder(
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu"
            )

    def _init_dense(self) -> None:
        cache_path = CACHE / "bge_m3_pure.npz"
        print("Loading dense model deepvk/USER-bge-m3...")
        self.dense = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")

        # Limit chunks per doc for speed
        c_texts, c_docs = [], []
        for did, text in zip(self.docs.doc_id, self.docs.text):
            chunks = chunk_text(str(text), 1000, 150)
            if len(chunks) > 8:
                idxs = np.linspace(0, len(chunks) - 1, 8).astype(int)
                chunks = [chunks[i] for i in idxs]
            for ch in chunks:
                c_texts.append(ch)
                c_docs.append(did)
        self.dense_chunk_doc = c_docs

        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self.chunk_emb = data["chunk_emb"]
            self.full_emb = data["full_emb"]
            print(f"loaded dense cache {cache_path}")
        else:
            print("encoding passages...")
            self.chunk_emb = self.dense.encode(
                c_texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.full_emb = self.dense.encode(
                [str(t)[:2000] for t in self.docs.text],
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            np.savez_compressed(
                cache_path, chunk_emb=self.chunk_emb, full_emb=self.full_emb
            )

        all_q = (
            self.train.question.astype(str).tolist()
            + self.test.question.astype(str).tolist()
        )
        uniq = list(dict.fromkeys(all_q))
        q_cache = CACHE / "bge_m3_queries.npz"
        if q_cache.exists():
            data = np.load(q_cache, allow_pickle=True)
            cached_q = list(data["questions"])
            if cached_q == uniq:
                for q, e in zip(cached_q, data["emb"]):
                    self.query_emb_cache[q] = e
                print("loaded query cache")
        if not self.query_emb_cache:
            print("encoding queries...")
            emb = self.dense.encode(
                uniq, batch_size=16, normalize_embeddings=True, show_progress_bar=True
            )
            for q, e in zip(uniq, emb):
                self.query_emb_cache[q] = e
            np.savez_compressed(
                q_cache, questions=np.array(uniq, dtype=object), emb=np.stack(list(emb))
            )

    def _q_emb(self, question: str) -> np.ndarray:
        e = self.query_emb_cache.get(question)
        if e is None:
            e = self.dense.encode([question], normalize_embeddings=True)[0]
            self.query_emb_cache[question] = e
        return e

    def bm25_rank(self, question: str, topn: int = 100) -> list[str]:
        q = tokenize_lemmas(question)
        sd = self.bm25_doc.get_scores(q)
        sc = self.bm25_chunk.get_scores(q)
        best: dict[str, float] = {}
        for i, s in enumerate(sc):
            d = self.chunk_doc[i]
            if d not in best or s > best[d]:
                best[d] = float(s)
        return rrf_fuse(
            [
                [self.doc_ids[i] for i in np.argsort(sd)[::-1][:topn]],
                [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:topn]],
            ]
        )

    def tfidf_rank(self, question: str, topn: int = 100) -> list[str]:
        sw = cosine_similarity(self.tf_word.transform([question]), self.D_word)[0]
        sc = cosine_similarity(self.tf_char.transform([question]), self.D_char)[0]
        return rrf_fuse(
            [
                [self.doc_ids[i] for i in np.argsort(sw)[::-1][:topn]],
                [self.doc_ids[i] for i in np.argsort(sc)[::-1][:topn]],
            ]
        )

    def dense_rank(self, question: str, topn: int = 100) -> list[str]:
        qe = self._q_emb(question)
        sims = self.chunk_emb @ qe
        best: dict[str, float] = {}
        for s, d in zip(sims, self.dense_chunk_doc):
            if d not in best or s > best[d]:
                best[d] = float(s)
        full = self.full_emb @ qe
        for i, s in enumerate(full):
            d = self.doc_ids[i]
            best[d] = max(best.get(d, -1.0), float(s))
        return [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:topn]]

    def first_stage(self, question: str, topn: int = CAND_N) -> list[str]:
        lists = [self.bm25_rank(question, 100), self.tfidf_rank(question, 100)]
        weights = [1.2, 0.8]
        if self.use_dense:
            lists.append(self.dense_rank(question, 100))
            weights.append(1.0)
        return rrf_fuse(lists, weights=weights)[:topn]

    def ce_rerank(self, question: str, cands: list[str], topk: int = TOP_K) -> list[str]:
        if not self.use_ce or self.ce is None or not cands:
            return cands[:topk]
        pairs = [(question, self.dmap[d][:2800]) for d in cands]
        scores = self.ce.predict(pairs, batch_size=16, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [cands[i] for i in order[:topk]]

    def retrieve(self, question: str, topk: int = TOP_K, rerank_n: int = 20) -> list[str]:
        cands = self.first_stage(question, max(CAND_N, rerank_n))
        return self.ce_rerank(question, cands[:rerank_n], topk=topk)

    def evaluate_train(self, use_ce: bool = True, ks=(1, 5, 10, 20)) -> dict:
        ranks = []
        for q, g in tqdm(
            zip(self.train.question.astype(str), self.train.gold_doc_id),
            total=len(self.train),
            desc="eval train",
        ):
            cands = self.first_stage(q, 50)
            if use_ce and self.use_ce:
                ranked = self.ce_rerank(q, cands[:20], topk=50)
                # append rest of first-stage for deep recall stats
                seen = set(ranked)
                ranked = ranked + [d for d in cands if d not in seen]
            else:
                ranked = cands
            ranks.append(ranked.index(g) + 1 if g in ranked else 999)
        out = {f"R@{k}": float(np.mean([r <= k for r in ranks])) for k in ks}
        print(out)
        return out

    def predict_submission(self, path: Path | None = None) -> pd.DataFrame:
        rows = []
        for _, row in tqdm(self.test.iterrows(), total=len(self.test), desc="predict"):
            for d in self.retrieve(str(row.question), TOP_K):
                rows.append({"qid": row.qid, "doc_id": d})
        sub = pd.DataFrame(rows)
        path = path or (OUT / "submission.csv")
        sub.to_csv(path, index=False)
        sub.to_csv(ROOT / "submission.csv", index=False)
        print(f"wrote {path} rows={len(sub)}")
        return sub


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--no-ce", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    r = PureRetriever(use_dense=not args.no_dense, use_ce=not args.no_ce)
    metrics = {}
    if not args.skip_eval:
        print("=== first-stage only ===")
        metrics["first_stage"] = r.evaluate_train(use_ce=False)
        if r.use_ce:
            print("=== with cross-encoder ===")
            metrics["ce"] = r.evaluate_train(use_ce=True)
    if args.eval_only:
        return
    r.predict_submission()
    with open(OUT / "metrics_pure.txt", "w") as f:
        for name, m in metrics.items():
            f.write(f"{name}: {m}\n")


if __name__ == "__main__":
    main()
