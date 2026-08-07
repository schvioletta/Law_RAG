"""Pure retrieval without train-answer leakage: BM25+RM3 + e5/BGE/e5-ft + LTR."""

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
    "суд",
    "судья",
    "истец",
    "ответчик",
    "дело",
    "г",
    "года",
    "руб",
    "рубль",
    "адрес",
    "фио",
    "представитель",
    "заседание",
    "протокол",
    "москва",
    "мещанский",
    "районный",
    "также",
    "данный",
    "настоящий",
    "указанный",
}

CAND = 70
TOP_K = 5
FEATS = [
    "bm25_doc",
    "bm25_chunk",
    "bm25_rm3_doc",
    "bm25_rm3_chunk",
    "rrf_inv",
    "tfidf_word",
    "tfidf_char",
    "e5_chunk",
    "e5_full",
    "bge_chunk",
    "bge_full",
    "ft_chunk",
    "ft_full",
    "overlap",
    "jaccard",
    "uniq_hits",
    "doc_len",
    "rank_bm",
    "rank_ft",
    "rank_bge",
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


def load_dense(path, docs, chunk_size, overlap, max_chunks):
    data = np.load(path)
    chunk_emb, full_emb = data["chunk_emb"], data["full_emb"]
    dens_docs = []
    for did, text in zip(docs.doc_id, docs.text):
        chs = chunk_text(str(text), chunk_size, overlap)
        if len(chs) > max_chunks:
            idxs = np.linspace(0, len(chs) - 1, max_chunks).astype(int)
            chs = [chs[i] for i in idxs]
        dens_docs.extend([did] * len(chs))
    assert len(dens_docs) == len(chunk_emb), (path, len(dens_docs), len(chunk_emb))
    return chunk_emb, full_emb, dens_docs


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
        for ch in chunk_text(str(text), 800, 160):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
    bm25c = BM25Okapi(chunk_toks)

    # collection term stats for RM3
    df = Counter()
    for toks in doc_toks:
        df.update(set(toks))
    N = len(doc_toks)

    cleaned = [" ".join(t) for t in doc_toks]
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000
    )
    Dc = tfc.fit_transform(docs.text.fillna(""))
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    print("load dense...")
    e5_chunk, e5_full, e5_docs = load_dense(
        CACHE / "e5_small_emb.npz", docs, 900, 150, 10
    )
    bge_chunk, bge_full, bge_docs = load_dense(
        CACHE / "bge_m3_pure.npz", docs, 1000, 150, 8
    )
    ft_chunk, ft_full, ft_docs = load_dense(
        CACHE / "e5_ft_emb.npz", docs, 900, 150, 12
    )

    allq = list(
        dict.fromkeys(
            train.question.astype(str).tolist() + test.question.astype(str).tolist()
        )
    )

    def qmap(path):
        qd = np.load(path, allow_pickle=True)
        assert list(qd["questions"]) == allq
        return {q: e for q, e in zip(qd["questions"], qd["emb"])}

    e5_q = qmap(CACHE / "e5_small_queries.npz")
    bge_q = qmap(CACHE / "bge_m3_queries.npz")
    ft_q = qmap(CACHE / "e5_ft_queries.npz")
    print("ready")

    def bm25_scores(q_tokens):
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
                [doc_ids[i] for i in np.argsort(sd)[::-1][:200]],
                [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:200]],
            ],
            weights=[1.0, 1.25],
        )
        return ranked, sd, best

    def rm3_expand(base_tokens, seed_docs, n_terms=18):
        """Pseudo-relevance feedback from top retrieved docs (no train labels)."""
        bag = Counter()
        for d in seed_docs[:4]:
            for t in doc_toks[id2i[d]]:
                if len(t) < 5:
                    continue
                # idf-weighted
                idf = np.log((N + 1) / (df[t] + 0.5))
                if idf < 1.2:
                    continue
                bag[t] += idf
        base = set(base_tokens)
        extra = [t for t, _ in bag.most_common(n_terms) if t not in base]
        return base_tokens + extra

    def dense_scores(qe, chunk_emb, full_emb, dens_docs):
        sims = chunk_emb @ qe
        best = {}
        for s, d in zip(sims, dens_docs):
            if d not in best or s > best[d]:
                best[d] = float(s)
        full = full_emb @ qe
        for i, s in enumerate(full):
            best[doc_ids[i]] = max(best.get(doc_ids[i], -1e9), float(s))
        ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:200]]
        return ranked, best, full

    def first_stage(q: str):
        base = tok(q)
        bm1, sd, best = bm25_scores(base)
        rm3 = rm3_expand(base, bm1)
        bm2, sd2, best2 = bm25_scores(rm3)
        e5r, e5m, e5f = dense_scores(e5_q[q], e5_chunk, e5_full, e5_docs)
        bger, bgem, bgef = dense_scores(bge_q[q], bge_chunk, bge_full, bge_docs)
        ftr, ftm, ftf = dense_scores(ft_q[q], ft_chunk, ft_full, ft_docs)
        sw = cosine_similarity(tfw.transform([" ".join(base)]), Dw)[0]
        sc = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:200]],
                [doc_ids[i] for i in np.argsort(sc)[::-1][:200]],
            ]
        )
        fused = rrf(
            [bm1, bm2, ftr, bger, e5r, tf],
            weights=[1.4, 1.2, 1.35, 1.1, 0.95, 0.85],
        )[:CAND]
        return {
            "cands": fused,
            "sd": sd,
            "best": best,
            "sd2": sd2,
            "best2": best2,
            "e5m": e5m,
            "e5f": e5f,
            "bgem": bgem,
            "bgef": bgef,
            "ftm": ftm,
            "ftf": ftf,
            "sw": sw,
            "sc": sc,
            "bm1": bm1,
            "ftr": ftr,
            "bger": bger,
        }

    def make_feats(q, pack):
        qt = set(tok(q))
        cands = pack["cands"]
        bm_rank = {d: r for r, d in enumerate(pack["bm1"])}
        ft_rank = {d: r for r, d in enumerate(pack["ftr"])}
        bge_rank = {d: r for r, d in enumerate(pack["bger"])}
        X = []
        for rank, d in enumerate(cands):
            i = id2i[d]
            dset = set(doc_toks[i])
            inter = len(qt & dset)
            union = len(qt | dset) + 1e-6
            uniq = sum(1 for t in qt if t in dset and len(t) >= 5)
            X.append(
                [
                    float(pack["sd"][i]),
                    float(pack["best"].get(d, 0)),
                    float(pack["sd2"][i]),
                    float(pack["best2"].get(d, 0)),
                    1.0 / (rank + 1),
                    float(pack["sw"][i]),
                    float(pack["sc"][i]),
                    float(pack["e5m"].get(d, 0)),
                    float(pack["e5f"][i]),
                    float(pack["bgem"].get(d, 0)),
                    float(pack["bgef"][i]),
                    float(pack["ftm"].get(d, 0)),
                    float(pack["ftf"][i]),
                    float(inter),
                    float(inter / union),
                    float(uniq),
                    float(lens[i]),
                    float(bm_rank.get(d, 400)),
                    float(ft_rank.get(d, 400)),
                    float(bge_rank.get(d, 400)),
                ]
            )
        return np.asarray(X, np.float32)

    print("features...")
    Xs, ys, gs = [], [], []
    # group by gold_doc for stricter CV
    doc_group = {d: i for i, d in enumerate(sorted(set(train.gold_doc_id)))}
    q_groups = []
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        pack = first_stage(q)
        cands = list(pack["cands"])
        if g not in cands:
            cands = cands[:-1] + [g]
        pack = dict(pack)
        pack["cands"] = cands
        Xs.append(make_feats(q, pack))
        ys.append(np.array([1 if d == g else 0 for d in cands], np.int32))
        gs.append(np.full(len(cands), qi, np.int32))
        q_groups.append(doc_group[g])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(gs)
    q_groups = np.array(q_groups)

    print("CV by gold_doc groups + honest eval...")
    gkf = GroupKFold(5)
    recalls_h, recalls_t, iters = [], [], []
    # split on question indices grouped by gold doc
    qidx = np.arange(len(train))
    for fold, (tr_q, te_q) in enumerate(gkf.split(qidx, groups=q_groups)):
        # map to feature rows
        tr_set, te_set = set(tr_q.tolist()), set(te_q.tolist())
        tr = np.array([i for i, g in enumerate(groups) if g in tr_set])
        te = np.array([i for i, g in enumerate(groups) if g in te_set])
        tr_g = [int((groups[tr] == g).sum()) for g in sorted(tr_set)]
        te_g = [int((groups[te] == g).sum()) for g in sorted(te_set)]
        # keep group order consistent with unique order
        tr_unique = []
        for g in groups[tr]:
            if g not in tr_unique:
                tr_unique.append(int(g))
        te_unique = []
        for g in groups[te]:
            if g not in te_unique:
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
                learning_rate=0.04,
                num_leaves=47,
                min_data_in_leaf=10,
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
        hits = nq = 0
        for g in te_unique:
            idx = np.where(groups[te] == g)[0]
            order = idx[np.argsort(-pred[idx])]
            if y[te][order][:TOP_K].sum() > 0:
                hits += 1
            nq += 1
        trainlike = hits / nq

        hits_h = 0
        for qi in te_q:
            row = train.iloc[int(qi)]
            q = str(row.question)
            gold = row.gold_doc_id
            pack = first_stage(q)
            scores = booster.predict(make_feats(q, pack))
            ltr = [pack["cands"][i] for i in np.argsort(-scores)]
            final = rrf([ltr, pack["cands"], pack["bm1"][:CAND]], weights=[1.7, 0.85, 0.7])[
                :TOP_K
            ]
            if gold in final:
                hits_h += 1
        honest = hits_h / len(te_q)
        print(
            f"fold {fold}: trainlike={trainlike:.4f} honest={honest:.4f} "
            f"iter={booster.best_iteration}"
        )
        recalls_t.append(trainlike)
        recalls_h.append(honest)
        iters.append(booster.best_iteration or 140)

    cv_h = float(np.mean(recalls_h))
    cv_t = float(np.mean(recalls_t))
    n_est = int(np.median(iters))
    print(f"CV honest={cv_h:.4f} trainlike={cv_t:.4f} rounds={n_est}")

    # final model on all data (groups by question id order)
    q_order = list(range(len(train)))
    group_sizes = [CAND] * len(train)
    booster = lgb.train(
        dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            learning_rate=0.04,
            num_leaves=47,
            min_data_in_leaf=10,
            feature_fraction=0.85,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbosity=-1,
            label_gain=list(range(2)),
        ),
        lgb.Dataset(X, y, group=group_sizes, feature_name=FEATS),
        num_boost_round=max(n_est, 150),
    )

    def retrieve(q: str):
        pack = first_stage(q)
        scores = booster.predict(make_feats(q, pack))
        ltr = [pack["cands"][i] for i in np.argsort(-scores)]
        return rrf([ltr, pack["cands"], pack["bm1"][:CAND]], weights=[1.7, 0.85, 0.7])[
            :TOP_K
        ]

    print("test...")
    rows = []
    for _, row in tqdm(test.iterrows(), total=len(test)):
        for d in retrieve(str(row.question)):
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    frac = float(sub.doc_id.isin(set(train.gold_doc_id)).mean())
    print(f"unique_docs={sub.doc_id.nunique()} train_gold_frac={frac:.3f} cv_honest={cv_h:.4f}")
    with open(OUT / "metrics_v5.txt", "w") as f:
        f.write(f"cv_honest_recall@5={cv_h:.6f}\n")
        f.write(f"cv_trainlike_recall@5={cv_t:.6f}\n")
        f.write(f"train_gold_frac={frac:.6f}\n")
        f.write(f"unique_docs={sub.doc_id.nunique()}\n")


if __name__ == "__main__":
    main()
