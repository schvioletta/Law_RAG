"""Hybrid retrieval + LightGBM LambdaRank for Law RAG."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from preprocess import chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

CAND_N = 50
TOP_K = 5
FEATURE_NAMES = [
    "bm25_doc",
    "bm25_chunk",
    "rrf_rank_inv",
    "tfidf_word",
    "tfidf_char",
    "tfidf_evidence",
    "tfidf_qbag",
    "dense_chunk",
    "dense_full",
    "dense_evidence",
    "token_overlap",
    "token_jaccard",
    "doc_len_z",
    "has_train_label",
    "n_train_q",
]


class LawRetriever:
    def __init__(self, use_dense: bool = True):
        self.use_dense = use_dense
        self.docs = pd.read_csv(DATA / "documents.csv")
        self.train = pd.read_csv(DATA / "train.csv")
        self.test = pd.read_csv(DATA / "test.csv")
        self.doc_ids = self.docs.doc_id.tolist()
        self.id2i = {d: i for i, d in enumerate(self.doc_ids)}

        self.doc_tokens = [
            tokenize_lemmas(t) for t in tqdm(self.docs.text, desc="tokenize docs")
        ]
        self.bm25_doc = BM25Okapi(self.doc_tokens)

        self.chunk_tokens: list[list[str]] = []
        self.chunk_doc: list[str] = []
        for did, text in tqdm(
            list(zip(self.docs.doc_id, self.docs.text)), desc="chunk docs"
        ):
            for ch in chunk_text(str(text), 1200, 200):
                self.chunk_tokens.append(tokenize_lemmas(ch))
                self.chunk_doc.append(did)
        self.bm25_chunk = BM25Okapi(self.chunk_tokens)

        self.tf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=80_000
        )
        self.D_word = self.tf_word.fit_transform(self.docs.text.fillna(""))
        self.tf_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 5), min_df=3, max_features=100_000
        )
        self.D_char = self.tf_char.fit_transform(self.docs.text.fillna(""))

        lens = np.array([len(str(t)) for t in self.docs.text], dtype=float)
        self.doc_len_z = (lens - lens.mean()) / (lens.std() + 1e-6)

        self.q_by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.ev_by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for _, row in self.train.iterrows():
            self.q_by_doc[row.gold_doc_id].append((row.qid, str(row.question)))
            self.ev_by_doc[row.gold_doc_id].append(
                (row.qid, str(row.gold_evidence_text))
            )

        self.doc_ev_text = [
            " ".join(t for _, t in self.ev_by_doc.get(d, [])) for d in self.doc_ids
        ]
        self.doc_q_text = [
            " ".join(t for _, t in self.q_by_doc.get(d, [])) for d in self.doc_ids
        ]
        self.n_train_q = np.array(
            [len(self.q_by_doc.get(d, [])) for d in self.doc_ids], dtype=float
        )

        self.tf_ev = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.D_ev = self.tf_ev.fit_transform(self.doc_ev_text)
        self.tf_qbag = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.D_qbag = self.tf_qbag.fit_transform(self.doc_q_text)

        # Precompute per-qid residual bags: full bag with that qid removed
        self.qid_residual_ev = {}
        self.qid_residual_q = {}
        for _, row in self.train.iterrows():
            d = row.gold_doc_id
            self.qid_residual_ev[row.qid] = " ".join(
                t for qid, t in self.ev_by_doc[d] if qid != row.qid
            )
            self.qid_residual_q[row.qid] = " ".join(
                t for qid, t in self.q_by_doc[d] if qid != row.qid
            )

        self.model = None
        self.dense = None
        self.chunk_emb = None
        self.chunk_doc_e: list[str] = []
        self.full_emb = None
        self.ev_emb = None
        self.ev_qids: list[str] = []
        self.ev_docs: list[str] = []
        self.query_emb_cache: dict[str, np.ndarray] = {}

        if self.use_dense:
            self._init_dense()

    def _init_dense(self) -> None:
        cache_path = CACHE / "e5_small_emb.npz"
        self.dense = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        chunk_texts: list[str] = []
        self.chunk_doc_e = []
        for did, text in zip(self.docs.doc_id, self.docs.text):
            chunks = chunk_text(str(text), 900, 150)
            if len(chunks) > 10:
                idxs = np.linspace(0, len(chunks) - 1, 10).astype(int)
                chunks = [chunks[i] for i in idxs]
            for ch in chunks:
                chunk_texts.append(ch)
                self.chunk_doc_e.append(did)

        self.ev_qids = self.train.qid.tolist()
        self.ev_docs = self.train.gold_doc_id.tolist()
        ev_texts = self.train.gold_evidence_text.fillna("").astype(str).tolist()

        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self.chunk_emb = data["chunk_emb"]
            self.full_emb = data["full_emb"]
            self.ev_emb = data["ev_emb"]
            print(f"loaded dense cache {cache_path}")
        else:
            print("encoding dense representations...")
            self.chunk_emb = self.dense.encode(
                ["passage: " + t for t in chunk_texts],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.full_emb = self.dense.encode(
                ["passage: " + str(t)[:1800] for t in self.docs.text],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            self.ev_emb = self.dense.encode(
                ["passage: " + t for t in ev_texts],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            np.savez_compressed(
                cache_path,
                chunk_emb=self.chunk_emb,
                full_emb=self.full_emb,
                ev_emb=self.ev_emb,
            )

        # Precompute embeddings for all known questions
        all_questions = (
            self.train.question.astype(str).tolist()
            + self.test.question.astype(str).tolist()
        )
        uniq = list(dict.fromkeys(all_questions))
        q_cache = CACHE / "e5_small_queries.npz"
        if q_cache.exists():
            data = np.load(q_cache, allow_pickle=True)
            cached_q = list(data["questions"])
            cached_e = data["emb"]
            if cached_q == uniq:
                for q, e in zip(cached_q, cached_e):
                    self.query_emb_cache[q] = e
                print("loaded query emb cache")
            else:
                q_cache.unlink(missing_ok=True)
        if not self.query_emb_cache:
            print("encoding queries...")
            emb = self.dense.encode(
                ["query: " + q for q in uniq],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for q, e in zip(uniq, emb):
                self.query_emb_cache[q] = e
            np.savez_compressed(
                q_cache, questions=np.array(uniq, dtype=object), emb=np.stack(emb)
            )

    def _bag_sims(self, question: str, exclude_qid: str | None = None):
        qq = self.tf_qbag.transform([question])
        qe = self.tf_ev.transform([question])
        sim_q = cosine_similarity(qq, self.D_qbag)[0].copy()
        sim_e = cosine_similarity(qe, self.D_ev)[0].copy()
        if exclude_qid is not None:
            gold = self.train.loc[self.train.qid == exclude_qid, "gold_doc_id"]
            if len(gold):
                gi = self.id2i[gold.iloc[0]]
                sim_q[gi] = float(
                    cosine_similarity(
                        qq, self.tf_qbag.transform([self.qid_residual_q[exclude_qid]])
                    )[0, 0]
                )
                sim_e[gi] = float(
                    cosine_similarity(
                        qe, self.tf_ev.transform([self.qid_residual_ev[exclude_qid]])
                    )[0, 0]
                )
        return sim_q, sim_e

    def _bm25_lists(self, question: str) -> tuple[np.ndarray, dict[str, float], list[str]]:
        q_tokens = tokenize_lemmas(question)
        sd = self.bm25_doc.get_scores(q_tokens)
        sc = self.bm25_chunk.get_scores(q_tokens)
        best_chunk: dict[str, float] = {}
        for i, s in enumerate(sc):
            d = self.chunk_doc[i]
            if d not in best_chunk or s > best_chunk[d]:
                best_chunk[d] = float(s)
        scores: dict[str, float] = defaultdict(float)
        for rank, idx in enumerate(np.argsort(sd)[::-1][:100]):
            scores[self.doc_ids[idx]] += 1.0 / (60 + rank + 1)
        for rank, (d, _) in enumerate(
            sorted(best_chunk.items(), key=lambda x: -x[1])[:100]
        ):
            scores[d] += 1.0 / (60 + rank + 1)
        lexical = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]
        return sd, best_chunk, lexical

    def first_stage(
        self, question: str, topn: int = CAND_N, exclude_qid: str | None = None
    ) -> tuple[list[str], np.ndarray, dict]:
        sd, best_chunk, lexical = self._bm25_lists(question)
        scores: dict[str, float] = {
            d: 1.0 / (60 + rank + 1) for rank, d in enumerate(lexical[:100])
        }
        sim_q, sim_e = self._bag_sims(question, exclude_qid=exclude_qid)
        for i, d in enumerate(self.doc_ids):
            scores[d] = scores.get(d, 0.0) + 0.40 * float(sim_q[i]) + 0.30 * float(
                sim_e[i]
            )
        if self.use_dense:
            dens_chunk, dens_full, dens_ev = self._dense_scores(
                question, exclude_qid=exclude_qid
            )
            dense_scores = {
                d: max(dens_chunk.get(d, 0.0), float(dens_full[i]), dens_ev.get(d, 0.0))
                for i, d in enumerate(self.doc_ids)
            }
            for rank, (d, _) in enumerate(
                sorted(dense_scores.items(), key=lambda x: -x[1])[:100]
            ):
                scores[d] = scores.get(d, 0.0) + 0.85 / (60 + rank + 1)

        ranked = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:topn]]
        return ranked, sd, best_chunk

    def _dense_scores(self, question: str, exclude_qid: str | None = None):
        q_emb = self.query_emb_cache.get(question)
        if q_emb is None:
            q_emb = self.dense.encode(
                ["query: " + question], normalize_embeddings=True
            )[0]
            self.query_emb_cache[question] = q_emb
        sims = self.chunk_emb @ q_emb
        dens_chunk: dict[str, float] = {}
        for s, d in zip(sims, self.chunk_doc_e):
            if d not in dens_chunk or s > dens_chunk[d]:
                dens_chunk[d] = float(s)
        dens_full = self.full_emb @ q_emb
        dens_ev: dict[str, float] = {}
        for s, d, qid in zip(self.ev_emb @ q_emb, self.ev_docs, self.ev_qids):
            if exclude_qid is not None and qid == exclude_qid:
                continue
            if d not in dens_ev or s > dens_ev[d]:
                dens_ev[d] = float(s)
        return dens_chunk, dens_full, dens_ev

    def features_for(
        self,
        question: str,
        cand_ids: list[str],
        sd: np.ndarray,
        best_chunk: dict,
        exclude_qid: str | None = None,
    ) -> np.ndarray:
        q_tokens = set(tokenize_lemmas(question))
        qw = self.tf_word.transform([question])
        qc = self.tf_char.transform([question])

        n_train = self.n_train_q.copy()
        if exclude_qid is not None:
            gold = self.train.loc[self.train.qid == exclude_qid, "gold_doc_id"]
            if len(gold):
                gi = self.id2i[gold.iloc[0]]
                n_train[gi] = max(0, n_train[gi] - 1)

        sw = cosine_similarity(qw, self.D_word)[0]
        sch = cosine_similarity(qc, self.D_char)[0]
        sq, se = self._bag_sims(question, exclude_qid=exclude_qid)

        dens_chunk = dens_full = dens_ev = {}
        if self.use_dense:
            dens_chunk, dens_full, dens_ev = self._dense_scores(
                question, exclude_qid=exclude_qid
            )

        feats = []
        for rank, d in enumerate(cand_ids):
            i = self.id2i[d]
            dset = set(self.doc_tokens[i])
            inter = len(q_tokens & dset)
            union = len(q_tokens | dset) + 1e-6
            feats.append(
                [
                    float(sd[i]),
                    float(best_chunk.get(d, 0.0)),
                    1.0 / (rank + 1),
                    float(sw[i]),
                    float(sch[i]),
                    float(se[i]),
                    float(sq[i]),
                    float(dens_chunk.get(d, 0.0)) if self.use_dense else 0.0,
                    float(dens_full[i]) if self.use_dense else 0.0,
                    float(dens_ev.get(d, 0.0)) if self.use_dense else 0.0,
                    float(inter),
                    float(inter / union),
                    float(self.doc_len_z[i]),
                    1.0 if n_train[i] > 0 else 0.0,
                    float(n_train[i]),
                ]
            )
        return np.asarray(feats, dtype=np.float32)

    def build_train_matrix(self):
        X_parts, y_parts, groups = [], [], []
        for qi, row in tqdm(
            self.train.iterrows(), total=len(self.train), desc="build features"
        ):
            q = str(row.question)
            g = row.gold_doc_id
            cands, sd, best_chunk = self.first_stage(q, CAND_N, exclude_qid=row.qid)
            if g not in cands:
                cands = cands[:-1] + [g]
            feats = self.features_for(
                q, cands, sd, best_chunk, exclude_qid=row.qid
            )
            labels = np.array([1 if d == g else 0 for d in cands], dtype=np.int32)
            X_parts.append(feats)
            y_parts.append(labels)
            groups.append(np.full(len(cands), qi, dtype=np.int32))
        return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(groups)

    def _train_booster(self, X, y, groups, num_boost_round=120, valid=None):
        group_sizes = [int(np.sum(groups == g)) for g in np.unique(groups)]
        dtrain = lgb.Dataset(
            X, label=y, group=group_sizes, feature_name=FEATURE_NAMES
        )
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5],
            "learning_rate": 0.05,
            "num_leaves": 47,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "label_gain": list(range(2)),
        }
        callbacks = []
        valid_sets = None
        if valid is not None:
            Xv, yv, gv = valid
            vg = [int(np.sum(gv == g)) for g in np.unique(gv)]
            dval = lgb.Dataset(
                Xv, label=yv, group=vg, reference=dtrain, feature_name=FEATURE_NAMES
            )
            valid_sets = [dval]
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
            num_boost_round = 400
        return lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )

    def cross_validate(self, n_splits: int = 5) -> float:
        X, y, groups = self.build_train_matrix()
        gkf = GroupKFold(n_splits=n_splits)
        recalls = []
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
            booster = self._train_booster(
                X[tr], y[tr], groups[tr], valid=(X[te], y[te], groups[te])
            )
            pred = booster.predict(X[te])
            hits = 0
            nq = 0
            for g in np.unique(groups[te]):
                idx = np.where(groups[te] == g)[0]
                order = idx[np.argsort(-pred[idx])]
                if y[te][order][:TOP_K].sum() > 0:
                    hits += 1
                nq += 1
            r = hits / nq
            print(
                f"fold {fold}: Recall@{TOP_K}={r:.4f} best_iter={booster.best_iteration}"
            )
            recalls.append(r)
        mean_r = float(np.mean(recalls))
        print(f"CV mean Recall@{TOP_K}={mean_r:.4f}")
        return mean_r

    def fit(self, num_boost_round: int = 200) -> lgb.Booster:
        X, y, groups = self.build_train_matrix()
        self.model = self._train_booster(X, y, groups, num_boost_round=num_boost_round)
        with open(OUT / "lgbm_ranker.pkl", "wb") as f:
            pickle.dump(self.model, f)
        imp = sorted(
            zip(FEATURE_NAMES, self.model.feature_importance("gain")),
            key=lambda x: -x[1],
        )
        print("feature importance (gain):")
        for name, val in imp:
            print(f"  {name:16s} {val}")
        return self.model

    def retrieve(self, question: str, topk: int = TOP_K) -> list[str]:
        cands, sd, best_chunk = self.first_stage(question, CAND_N)
        feats = self.features_for(question, cands, sd, best_chunk)
        scores = self.model.predict(feats)
        ltr_ranked = [cands[i] for i in np.argsort(-scores)]

        sim_q, sim_e = self._bag_sims(question)
        bag_conf = float(max(sim_q.max(), sim_e.max()))

        # High bag confidence → trust LambdaRank (train-doc neighborhood).
        # Low confidence → fuse with lexical/dense so unseen docs can surface.
        if bag_conf >= 0.28:
            return ltr_ranked[:topk]

        _, _, lexical = self._bm25_lists(question)
        fused: dict[str, float] = defaultdict(float)
        for rank, d in enumerate(ltr_ranked[:40]):
            fused[d] += 1.5 / (60 + rank + 1)
        for rank, d in enumerate(lexical[:40]):
            fused[d] += 1.2 / (60 + rank + 1)
        if self.use_dense:
            dens_chunk, dens_full, dens_ev = self._dense_scores(question)
            dense_scores = {
                d: max(dens_chunk.get(d, 0.0), float(dens_full[i]), dens_ev.get(d, 0.0))
                for i, d in enumerate(self.doc_ids)
            }
            for rank, (d, _) in enumerate(
                sorted(dense_scores.items(), key=lambda x: -x[1])[:40]
            ):
                fused[d] += 1.0 / (60 + rank + 1)
        return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:topk]]

    def predict_submission(self, path: Path | None = None) -> pd.DataFrame:
        rows = []
        for _, row in tqdm(self.test.iterrows(), total=len(self.test), desc="predict"):
            for d in self.retrieve(str(row.question), TOP_K):
                rows.append({"qid": row.qid, "doc_id": d})
        sub = pd.DataFrame(rows)
        path = path or (OUT / "submission.csv")
        sub.to_csv(path, index=False)
        # also copy to repo root for convenience
        sub.to_csv(ROOT / "submission.csv", index=False)
        print(f"wrote {path} and {ROOT / 'submission.csv'} rows={len(sub)}")
        return sub

    def train_recall(self) -> float:
        hits = 0
        for _, row in tqdm(self.train.iterrows(), total=len(self.train), desc="train R@5"):
            if row.gold_doc_id in self.retrieve(str(row.question), TOP_K):
                hits += 1
        r = hits / len(self.train)
        print(f"Train Recall@{TOP_K}={r:.4f}")
        return r


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    retriever = LawRetriever(use_dense=not args.no_dense)
    cv = None
    if not args.skip_cv:
        cv = retriever.cross_validate()
        if args.cv_only:
            with open(OUT / "metrics.txt", "w") as f:
                f.write(f"cv_recall@5={cv:.6f}\n")
            return
    retriever.fit()
    train_r = retriever.train_recall()
    retriever.predict_submission()
    with open(OUT / "metrics.txt", "w") as f:
        if cv is not None:
            f.write(f"cv_recall@5={cv:.6f}\n")
        f.write(f"train_recall@5={train_r:.6f}\n")


if __name__ == "__main__":
    main()
