"""v10: quick FT e5 on q↔evidence (2 epochs) + BM25/TFIDF + LightGBM LTR.

No CE (hurt holdout). No train-gold bag. Honest holdout by gold_doc.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
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
BASE = "intfloat/multilingual-e5-small"
TOP_K, CAND = 5, 80
SEED = 42
CHUNK_SIZE, CHUNK_OV = 800, 120

FEATS = [
    "bm25_doc", "bm25_chunk", "rrf_inv", "tfidf_word", "tfidf_char",
    "ft_chunk", "ft_full", "overlap", "jaccard", "uniq_hits", "long_hits",
    "doc_len", "rank_bm", "rank_ft",
]


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
    id2i = {d: i for i, d in enumerate(doc_ids)}
    dmap = dict(zip(docs.doc_id, docs.text.astype(str)))

    print("BM25/TFIDF...")
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
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000)
    Dc = tfc.fit_transform(docs.text.fillna(""))
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    # Fine-tune e5 on evidence
    print("FT e5 on evidence...")
    model = SentenceTransformer(BASE)
    examples = []
    for _, row in train.iterrows():
        q = str(row.question)
        ev = str(row.gold_evidence_text)[:1600]
        examples.append(InputExample(texts=[f"query: {q}", f"passage: {ev}"]))
        # window from doc around evidence
        doc = dmap[row.gold_doc_id]
        idx = doc.find(ev[:60]) if len(ev) >= 60 else -1
        if idx >= 0:
            w = doc[max(0, idx - 50) : idx + min(len(ev), 900) + 50]
            examples.append(InputExample(texts=[f"query: {q}", f"passage: {w}"]))
    random.shuffle(examples)
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=3,
        warmup_steps=max(20, len(loader) // 5),
        output_path=str(OUT / "e5_evid_v10"),
        show_progress_bar=True,
        use_amp=False,
        optimizer_params={"lr": 2e-5},
    )
    model = SentenceTransformer(str(OUT / "e5_evid_v10"))

    print("encode corpus...")
    chunk_emb = model.encode(
        [f"passage: {c}" for c in chunk_texts],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    full_emb = model.encode(
        [f"passage: {t}" for t in docs.text.astype(str)],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.savez_compressed(CACHE / "e5_evid_v10.npz", chunk_emb=chunk_emb, full_emb=full_emb)

    all_q = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    all_qe = model.encode(
        [f"query: {q}" for q in all_q],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    train_qe, test_qe = all_qe[: len(train)], all_qe[len(train) :]

    def dense_best(qe):
        best = {}
        for s, d in zip(chunk_emb @ qe, chunk_doc):
            if d not in best or s > best[d]:
                best[d] = float(s)
        full = full_emb @ qe
        for i, s in enumerate(full):
            best[doc_ids[i]] = max(best.get(doc_ids[i], -1e9), float(s))
        ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])]
        return ranked, best, full

    def first_stage(q, qe, n=CAND):
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
        dens_rank, dens_map, dens_full = dense_best(qe)
        sw = cosine_similarity(tfw.transform([" ".join(tok(q))]), Dw)[0]
        sch = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf_rank = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:220]],
                [doc_ids[i] for i in np.argsort(sch)[::-1][:220]],
            ]
        )
        fused = rrf(
            [bm_doc, bm_ch, dens_rank[:220], tf_rank],
            weights=[1.35, 1.45, 1.6, 0.8],
        )[:n]
        return {
            "cands": fused,
            "sd": sd,
            "best_ch": best_ch,
            "dens_map": dens_map,
            "dens_full": dens_full,
            "sw": sw,
            "sc": sch,
            "bm_doc": bm_doc,
            "dens_rank": dens_rank,
        }

    # first-stage metrics
    hits = defaultdict(int)
    for i, row in tqdm(train.iterrows(), total=len(train), desc="fs"):
        pack = first_stage(str(row.question), train_qe[i], n=50)
        for k in (1, 5, 10, 20, 50):
            hits[k] += row.gold_doc_id in pack["cands"][:k]
    print("first-stage", {k: round(hits[k] / len(train), 4) for k in sorted(hits)})

    def make_feats(q, pack):
        qt = set(tok(q))
        bm_rank = {d: r for r, d in enumerate(pack["bm_doc"])}
        ft_rank = {d: r for r, d in enumerate(pack["dens_rank"])}
        X = []
        for rank, d in enumerate(pack["cands"]):
            i = id2i[d]
            dset = set(doc_toks[i])
            inter = len(qt & dset)
            union = len(qt | dset) + 1e-6
            X.append(
                [
                    float(pack["sd"][i]),
                    float(pack["best_ch"].get(d, 0)),
                    1.0 / (rank + 1),
                    float(pack["sw"][i]),
                    float(pack["sc"][i]),
                    float(pack["dens_map"].get(d, 0)),
                    float(pack["dens_full"][i]),
                    float(inter),
                    float(inter / union),
                    float(sum(1 for t in qt if t in dset and len(t) >= 5)),
                    float(sum(1 for t in qt if t in dset and len(t) >= 7)),
                    float(lens[i]),
                    float(bm_rank.get(d, 400)),
                    float(ft_rank.get(d, 400)),
                ]
            )
        return np.asarray(X, np.float32)

    print("build LTR features...")
    Xs, ys, gs, q_groups = [], [], [], []
    doc_group = {d: i for i, d in enumerate(sorted(set(train.gold_doc_id)))}
    packs_train = []
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        pack = first_stage(q, train_qe[qi], n=CAND)
        cands = list(pack["cands"])
        if g not in cands:
            cands = cands[:-1] + [g]
        pack = dict(pack)
        pack["cands"] = cands
        packs_train.append(pack)
        Xs.append(make_feats(q, pack))
        ys.append(np.array([1 if d == g else 0 for d in cands], np.int32))
        gs.append(np.full(len(cands), qi, np.int32))
        q_groups.append(doc_group[g])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    q_groups = np.array(q_groups)

    print("CV LTR...")
    gkf = GroupKFold(5)
    qidx = np.arange(len(train))
    recalls, iters = [], []
    for fold, (tr_q, te_q) in enumerate(gkf.split(qidx, groups=q_groups)):
        tr_set, te_set = set(tr_q.tolist()), set(te_q.tolist())
        tr = np.array([i for i, g in enumerate(groups) if g in tr_set])
        te = np.array([i for i, g in enumerate(groups) if g in te_set])
        tr_unique, te_unique = [], []
        for g in groups[tr]:
            if int(g) not in tr_unique:
                tr_unique.append(int(g))
        for g in groups[te]:
            if int(g) not in te_unique:
                te_unique.append(int(g))
        tr_g = [int((groups[tr] == g).sum()) for g in tr_unique]
        te_g = [int((groups[te] == g).sum()) for g in te_unique]
        dtr = lgb.Dataset(X[tr], y[tr], group=tr_g, feature_name=FEATS)
        dve = lgb.Dataset(X[te], y[te], group=te_g, reference=dtr, feature_name=FEATS)
        booster = lgb.train(
            dict(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[5],
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=15,
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=1,
                verbosity=-1,
                label_gain=list(range(2)),
            ),
            dtr,
            num_boost_round=400,
            valid_sets=[dve],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        hit = 0
        for qi in te_q:
            row = train.iloc[int(qi)]
            pack = first_stage(str(row.question), train_qe[int(qi)], n=CAND)
            scores = booster.predict(make_feats(str(row.question), pack))
            ltr = [pack["cands"][i] for i in np.argsort(-scores)]
            final = rrf(
                [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["dens_rank"][:CAND]],
                weights=[1.9, 0.7, 0.8, 1.1],
            )[:TOP_K]
            hit += row.gold_doc_id in final
        score = hit / len(te_q)
        print(f"fold {fold}: honest={score:.4f} iter={booster.best_iteration}")
        recalls.append(score)
        iters.append(booster.best_iteration or 120)

    cv = float(np.mean(recalls))
    n_est = int(np.median(iters))
    print(f"CV honest={cv:.4f} rounds={n_est}")

    booster = lgb.train(
        dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=15,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            verbosity=-1,
            label_gain=list(range(2)),
        ),
        lgb.Dataset(X, y, group=[CAND] * len(train), feature_name=FEATS),
        num_boost_round=max(n_est, 140),
    )

    rows = []
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        q = str(row.question)
        pack = first_stage(q, test_qe[i], n=CAND)
        scores = booster.predict(make_feats(q, pack))
        ltr = [pack["cands"][j] for j in np.argsort(-scores)]
        final = rrf(
            [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["dens_rank"][:CAND]],
            weights=[1.9, 0.7, 0.8, 1.1],
        )[:TOP_K]
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    metrics = {
        "cv_honest": cv,
        "folds": recalls,
        "fs": {str(k): hits[k] / len(train) for k in sorted(hits)},
        "unique_docs": int(sub.doc_id.nunique()),
        "train_gold_frac": float(sub.doc_id.isin(set(train.gold_doc_id)).mean()),
    }
    print(metrics)
    with open(OUT / "metrics_v10.txt", "w") as f:
        f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
