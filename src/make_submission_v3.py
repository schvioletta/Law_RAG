"""BGE-m3 + expanded BM25 + LightGBM (no CE — CE hurt Recall@5)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from preprocess import STOPWORDS, chunk_text, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"

LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "исковой", "заявление", "решение",
    "определение", "дело", "г", "года", "руб", "рубль", "адрес", "фио",
    "представитель", "третье", "лицо", "заседание", "протокол", "статья",
    "гражданский", "процессуальный", "кодекс", "рф", "москва", "мещанский",
    "районный", "апелляционный", "кассационный", "инстанция", "жалоб",
    "жалоба", "удовлетворить", "отказать", "взыскать", "сумма", "размер",
    "также", "данный", "настоящий", "указанный", "согласно", "который",
    "которая", "которое", "являться", "иметь", "мочь",
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
    "uniq_hits",
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
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    id2i = {d: i for i, d in enumerate(doc_ids)}

    print("tokenize...")
    doc_toks = [tok(t) for t in tqdm(docs.text)]
    bm25d = BM25Okapi(doc_toks)
    chunk_toks, chunk_doc = [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 900, 180):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
    bm25c = BM25Okapi(chunk_toks)

    cleaned = [" ".join(t) for t in doc_toks]
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000
    )
    Dc = tfc.fit_transform(docs.text.fillna(""))
    q_tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Qmat = q_tf.fit_transform(train.question.astype(str))
    train_ev = [tok(str(t)) for t in train.gold_evidence_text]
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    data = np.load(CACHE / "bge_m3_pure.npz")
    chunk_emb, full_emb = data["chunk_emb"], data["full_emb"]
    dens_docs = []
    for did, text in zip(docs.doc_id, docs.text):
        chs = chunk_text(str(text), 1000, 150)
        if len(chs) > 8:
            idxs = np.linspace(0, len(chs) - 1, 8).astype(int)
            chs = [chs[i] for i in idxs]
        dens_docs.extend([did] * len(chs))
    assert len(dens_docs) == len(chunk_emb)

    allq = list(
        dict.fromkeys(
            train.question.astype(str).tolist() + test.question.astype(str).tolist()
        )
    )
    qd = np.load(CACHE / "bge_m3_queries.npz", allow_pickle=True)
    assert list(qd["questions"]) == allq
    qemb = {q: e for q, e in zip(qd["questions"], qd["emb"])}
    print("indexes ready")

    def expand(q: str, n=8, add=14) -> list[str]:
        base = tok(q)
        sims = cosine_similarity(q_tf.transform([q]), Qmat)[0]
        bag = Counter()
        used = 0
        for j in np.argsort(-sims):
            if sims[j] < 0.12:
                break
            if str(train.iloc[j].question) == q:
                continue
            for t in train_ev[j]:
                if len(t) >= 5:
                    bag[t] += 1 + float(sims[j])
            used += 1
            if used >= n:
                break
        return base + [t for t, _ in bag.most_common(add) if t not in set(base)]

    def bm25_list(q_tokens, topn=150):
        qw = weighted(q_tokens)
        sd = bm25d.get_scores(qw)
        sc = bm25c.get_scores(qw)
        best = {}
        for i, s in enumerate(sc):
            d = chunk_doc[i]
            if d not in best or s > best[d]:
                best[d] = float(s)
        ranked = rrf(
            [
                [doc_ids[i] for i in np.argsort(sd)[::-1][:topn]],
                [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:topn]],
            ],
            weights=[1.0, 1.15],
        )
        return ranked, sd, best

    def dense_list(q, topn=150):
        qe = qemb[q]
        sims = chunk_emb @ qe
        best = {}
        for s, d in zip(sims, dens_docs):
            if d not in best or s > best[d]:
                best[d] = float(s)
        full = full_emb @ qe
        for i, s in enumerate(full):
            best[doc_ids[i]] = max(best.get(doc_ids[i], -1), float(s))
        ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:topn]]
        return ranked, best, full

    def first_stage(q: str):
        base = tok(q)
        exp = expand(q)
        bm1, sd, best = bm25_list(base)
        bm2, sd2, best2 = bm25_list(exp)
        for d, s in best2.items():
            best[d] = max(best.get(d, 0), s)
        sd = np.maximum(sd, sd2)
        dens, dens_map, full = dense_list(q)
        sw = cosine_similarity(tfw.transform([" ".join(base)]), Dw)[0]
        sc = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:150]],
                [doc_ids[i] for i in np.argsort(sc)[::-1][:150]],
            ]
        )
        fused = rrf([bm1, bm2, dens, tf], weights=[1.35, 1.05, 1.25, 0.75])[:CAND]
        return fused, sd, best, dens_map, full, sw, sc

    def make_feats(q, cands, sd, best, dens, full, sw, sc):
        qt = set(tok(q))
        X = []
        for rank, d in enumerate(cands):
            i = id2i[d]
            dset = set(doc_toks[i])
            inter = len(qt & dset)
            union = len(qt | dset) + 1e-6
            uniq = sum(1 for t in qt if t in dset and len(t) >= 5)
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
                    float(uniq),
                ]
            )
        return np.asarray(X, np.float32)

    print("build features...")
    Xs, ys, gs = [], [], []
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        cands, sd, best, dens, full, sw, sc = first_stage(q)
        if g not in cands:
            cands = cands[:-1] + [g]
        Xs.append(make_feats(q, cands, sd, best, dens, full, sw, sc))
        ys.append(np.array([1 if d == g else 0 for d in cands], np.int32))
        gs.append(np.full(len(cands), qi, np.int32))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)

    print("CV...")
    gkf = GroupKFold(5)
    recalls, iters = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        tr_g = [int((groups[tr] == g).sum()) for g in np.unique(groups[tr])]
        te_g = [int((groups[te] == g).sum()) for g in np.unique(groups[te])]
        dtr = lgb.Dataset(X[tr], y[tr], group=tr_g, feature_name=FEATS)
        dve = lgb.Dataset(X[te], y[te], group=te_g, reference=dtr, feature_name=FEATS)
        booster = lgb.train(
            dict(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[5],
                learning_rate=0.04,
                num_leaves=63,
                min_data_in_leaf=10,
                feature_fraction=0.9,
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
        hits = nq = 0
        for g in np.unique(groups[te]):
            idx = np.where(groups[te] == g)[0]
            order = idx[np.argsort(-pred[idx])]
            if y[te][order][:TOP_K].sum() > 0:
                hits += 1
            nq += 1
        print(f"fold {fold}: R@5={hits/nq:.4f} iter={booster.best_iteration}")
        recalls.append(hits / nq)
        iters.append(booster.best_iteration or 120)
    cv = float(np.mean(recalls))
    n_est = int(np.median(iters))
    print(f"CV mean Recall@5={cv:.4f}")

    booster = lgb.train(
        dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            learning_rate=0.04,
            num_leaves=63,
            min_data_in_leaf=10,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbosity=-1,
            label_gain=list(range(2)),
        ),
        lgb.Dataset(X, y, group=[CAND] * len(train), feature_name=FEATS),
        num_boost_round=max(n_est, 140),
    )

    def retrieve(q: str):
        cands, sd, best, dens, full, sw, sc = first_stage(q)
        scores = booster.predict(make_feats(q, cands, sd, best, dens, full, sw, sc))
        ltr = [cands[i] for i in np.argsort(-scores)]
        return rrf([ltr, cands], weights=[1.6, 0.9])[:TOP_K]

    print("predict test...")
    rows = []
    for _, row in tqdm(test.iterrows(), total=len(test)):
        for d in retrieve(str(row.question)):
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    frac = float(sub.doc_id.isin(set(train.gold_doc_id)).mean())
    print(f"unique_docs={sub.doc_id.nunique()} train_gold_frac={frac:.3f} cv={cv:.4f}")
    with open(OUT / "metrics_v3.txt", "w") as f:
        f.write(f"cv_recall@5={cv:.6f}\n")
        f.write(f"train_gold_frac={frac:.6f}\n")


if __name__ == "__main__":
    main()
