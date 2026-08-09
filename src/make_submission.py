"""Generate competition submission with pure retrieval (no train-doc leakage)."""

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

from preprocess import STOPWORDS, chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

# Extra legal boilerplate that hurts discrimination
LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "исковой", "заявление", "решение",
    "определение", "дело", "г", "года", "руб", "рубль", "адрес", "фио",
    "представитель", "третье", "лицо", "заседание", "протокол", "статья",
    "гражданский", "процессуальный", "кодекс", "рф", "москва", "мещанский",
    "районный", "апелляционный", "кассационный", "инстанция", "жалоб",
    "жалоба", "удовлетворить", "отказать", "взыскать", "сумма", "размер",
}

CAND = 50
TOP_K = 5
FEATS = [
    "bm25_doc",
    "bm25_chunk",
    "rrf_inv",
    "tfidf_word",
    "tfidf_char",
    "dense_chunk",
    "dense_full",
    "overlap",
    "jaccard",
    "doc_len",
    "unique_q_hits",
]


def tok(text: str) -> list[str]:
    return [t for t in tokenize_lemmas(text) if t not in LEGAL_STOP]


def main():
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    id2i = {d: i for i, d in enumerate(doc_ids)}

    print("tokenizing...")
    doc_toks = [tok(t) for t in tqdm(docs.text)]
    bm25d = BM25Okapi(doc_toks)

    chunk_toks, chunk_doc = [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 1000, 200):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
    bm25c = BM25Okapi(chunk_toks)

    # Cleaned docs for TF-IDF
    cleaned = [" ".join(t) for t in doc_toks]
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=120_000
    )
    Dc = tfc.fit_transform(docs.text.fillna(""))
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    print("dense e5...")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    c_texts, c_docs = [], []
    for did, text in zip(docs.doc_id, docs.text):
        chs = chunk_text(str(text), 900, 150)
        if len(chs) > 10:
            idxs = np.linspace(0, len(chs) - 1, 10).astype(int)
            chs = [chs[i] for i in idxs]
        for ch in chs:
            c_texts.append(ch)
            c_docs.append(did)

    cache = CACHE / "e5_small_emb.npz"
    data = np.load(cache)
    chunk_emb, full_emb = data["chunk_emb"], data["full_emb"]

    allq = list(
        dict.fromkeys(
            train.question.astype(str).tolist() + test.question.astype(str).tolist()
        )
    )
    qcache = CACHE / "e5_small_queries.npz"
    qdata = np.load(qcache, allow_pickle=True)
    if list(qdata["questions"]) == allq:
        qemb = {q: e for q, e in zip(qdata["questions"], qdata["emb"])}
    else:
        emb = model.encode(
            ["query: " + q for q in allq],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        qemb = {q: e for q, e in zip(allq, emb)}

    def first_stage(q: str):
        qt = tok(q)
        # upweight rare content words by repeating nouns-ish long tokens
        qt_w = []
        for t in qt:
            qt_w.append(t)
            if len(t) >= 6:
                qt_w.append(t)
        sd = bm25d.get_scores(qt_w)
        sc = bm25c.get_scores(qt_w)
        best = {}
        for i, s in enumerate(sc):
            d = chunk_doc[i]
            if d not in best or s > best[d]:
                best[d] = float(s)
        scores = defaultdict(float)
        for r, i in enumerate(np.argsort(sd)[::-1][:120]):
            scores[doc_ids[i]] += 1.2 / (60 + r + 1)
        for r, (d, _) in enumerate(sorted(best.items(), key=lambda x: -x[1])[:120]):
            scores[d] += 1.2 / (60 + r + 1)

        qe = qemb[q]
        sims = chunk_emb @ qe
        dens = {}
        for s, d in zip(sims, c_docs):
            if d not in dens or s > dens[d]:
                dens[d] = float(s)
        full = full_emb @ qe
        for i, s in enumerate(full):
            dens[doc_ids[i]] = max(dens.get(doc_ids[i], -1), float(s))
        for r, (d, _) in enumerate(sorted(dens.items(), key=lambda x: -x[1])[:120]):
            scores[d] += 1.0 / (60 + r + 1)

        ranked = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:CAND]]
        return ranked, sd, best, dens, full

    def make_feats(q, cands, sd, best, dens, full):
        qt = set(tok(q))
        sw = cosine_similarity(tfw.transform([" ".join(tok(q))]), Dw)[0]
        sc = cosine_similarity(tfc.transform([q]), Dc)[0]
        X = []
        for rank, d in enumerate(cands):
            i = id2i[d]
            dset = set(doc_toks[i])
            inter = len(qt & dset)
            union = len(qt | dset) + 1e-6
            # how many distinctive query terms appear
            uniq_hits = sum(1 for t in qt if t in dset and len(t) >= 5)
            X.append(
                [
                    float(sd[i]),
                    float(best.get(d, 0)),
                    1 / (rank + 1),
                    float(sw[i]),
                    float(sc[i]),
                    float(dens.get(d, 0)),
                    float(full[i]),
                    float(inter),
                    float(inter / union),
                    float(lens[i]),
                    float(uniq_hits),
                ]
            )
        return np.asarray(X, np.float32)

    print("building features...")
    Xs, ys, gs, cand_lists = [], [], [], []
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        cands, sd, best, dens, full = first_stage(q)
        if g not in cands:
            cands = cands[:-1] + [g]
        Xs.append(make_feats(q, cands, sd, best, dens, full))
        ys.append(np.array([1 if d == g else 0 for d in cands], np.int32))
        gs.append(np.full(len(cands), qi, np.int32))
        cand_lists.append(cands)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)

    print("CV...")
    gkf = GroupKFold(5)
    recalls = []
    best_iters = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        tr_g = [int((groups[tr] == g).sum()) for g in np.unique(groups[tr])]
        te_g = [int((groups[te] == g).sum()) for g in np.unique(groups[te])]
        dtr = lgb.Dataset(X[tr], label=y[tr], group=tr_g, feature_name=FEATS)
        dve = lgb.Dataset(
            X[te], label=y[te], group=te_g, reference=dtr, feature_name=FEATS
        )
        booster = lgb.train(
            dict(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[5],
                learning_rate=0.05,
                num_leaves=63,
                min_data_in_leaf=12,
                feature_fraction=0.85,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbosity=-1,
                label_gain=list(range(2)),
            ),
            dtr,
            num_boost_round=500,
            valid_sets=[dve],
            callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
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
        print(f"fold {fold}: R@5={hits/nq:.4f} iter={booster.best_iteration}")
        recalls.append(hits / nq)
        best_iters.append(booster.best_iteration or 100)
    cv = float(np.mean(recalls))
    n_est = int(np.median(best_iters))
    print(f"CV mean Recall@5={cv:.4f}, rounds={n_est}")

    booster = lgb.train(
        dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=12,
            feature_fraction=0.85,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbosity=-1,
            label_gain=list(range(2)),
        ),
        lgb.Dataset(X, label=y, group=[CAND] * len(train), feature_name=FEATS),
        num_boost_round=max(n_est, 120),
    )
    with open(OUT / "lgbm_pure.pkl", "wb") as f:
        pickle.dump(booster, f)

    def retrieve(q: str):
        cands, sd, best, dens, full = first_stage(q)
        scores = booster.predict(make_feats(q, cands, sd, best, dens, full))
        return [cands[i] for i in np.argsort(-scores)[:TOP_K]]

    # train recall (optimistic fit)
    hits = sum(
        row.gold_doc_id in retrieve(str(row.question))
        for _, row in tqdm(train.iterrows(), total=len(train), desc="train R@5")
    )
    train_r = hits / len(train)
    print(f"Train Recall@5={train_r:.4f}")

    rows = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        for d in retrieve(str(row.question)):
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)
    frac = sub.doc_id.isin(set(train.gold_doc_id)).mean()
    print(
        f"submission docs={sub.doc_id.nunique()} train_gold_frac={frac:.3f} rows={len(sub)}"
    )
    with open(OUT / "metrics_pure.txt", "w") as f:
        f.write(f"cv_recall@5={cv:.6f}\n")
        f.write(f"train_recall@5={train_r:.6f}\n")
        f.write(f"train_gold_frac={frac:.6f}\n")


if __name__ == "__main__":
    main()
