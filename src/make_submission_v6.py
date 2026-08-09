"""Max-chunk FT dense + BM25/BGE + soft paraphrase boost + LTR (no answer-leak)."""

from __future__ import annotations

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

CAND = 80
TOP_K = 5
FEATS = [
    "bm25_doc",
    "bm25_chunk",
    "rrf_inv",
    "tfidf_word",
    "tfidf_char",
    "bge_chunk",
    "bge_full",
    "ft_chunk",
    "ft_full",
    "ft_dense_all",
    "overlap",
    "jaccard",
    "uniq_hits",
    "long_hits",
    "doc_len",
    "rank_bm",
    "rank_ft",
    "rank_bge",
    "nn_boost",
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
        for ch in chunk_text(str(text), 700, 140):
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
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    # question similarity for soft paraphrase boost
    q_tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Qmat = q_tf.fit_transform(train.question.astype(str))
    train_gold = train.gold_doc_id.tolist()

    print("dense caches + full FT chunk index...")
    bge = np.load(CACHE / "bge_m3_pure.npz")
    bge_chunk, bge_full = bge["chunk_emb"], bge["full_emb"]
    bge_docs = []
    for did, text in zip(docs.doc_id, docs.text):
        chs = chunk_text(str(text), 1000, 150)
        if len(chs) > 8:
            idxs = np.linspace(0, len(chs) - 1, 8).astype(int)
            chs = [chs[i] for i in idxs]
        bge_docs.extend([did] * len(chs))

    ft = np.load(CACHE / "e5_ft_emb.npz")
    ft_chunk, ft_full = ft["chunk_emb"], ft["full_emb"]
    ft_docs = []
    for did, text in zip(docs.doc_id, docs.text):
        chs = chunk_text(str(text), 900, 150)
        if len(chs) > 12:
            idxs = np.linspace(0, len(chs) - 1, 12).astype(int)
            chs = [chs[i] for i in idxs]
        ft_docs.extend([did] * len(chs))

    # denser FT chunk index for reranking
    dense_cache = CACHE / "e5_ft_dense_all.npz"
    all_chunks, all_chunk_docs = [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 650, 120):
            all_chunks.append(ch)
            all_chunk_docs.append(did)
    if dense_cache.exists():
        dd = np.load(dense_cache)
        assert len(dd["emb"]) == len(all_chunks)
        dense_all = dd["emb"]
    else:
        model = SentenceTransformer(str(OUT / "e5_ft"), device="cpu")
        dense_all = model.encode(
            ["passage: " + c for c in all_chunks],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.savez(dense_cache, emb=dense_all)

    allq = list(
        dict.fromkeys(
            train.question.astype(str).tolist() + test.question.astype(str).tolist()
        )
    )

    def qmap(path):
        qd = np.load(path, allow_pickle=True)
        assert list(qd["questions"]) == allq
        return {q: e for q, e in zip(qd["questions"], qd["emb"])}

    bge_q = qmap(CACHE / "bge_m3_queries.npz")
    ft_q = qmap(CACHE / "e5_ft_queries.npz")
    print("ready", "dense_all", len(all_chunks))

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
                [doc_ids[i] for i in np.argsort(sd)[::-1][:220]],
                [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:220]],
            ],
            weights=[1.0, 1.3],
        )
        return ranked, sd, best

    def dense_scores(qe, chunk_emb, full_emb, dens_docs):
        sims = chunk_emb @ qe
        best = {}
        for s, d in zip(sims, dens_docs):
            if d not in best or s > best[d]:
                best[d] = float(s)
        full = full_emb @ qe
        for i, s in enumerate(full):
            best[doc_ids[i]] = max(best.get(doc_ids[i], -1e9), float(s))
        ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:220]]
        return ranked, best, full

    def dense_all_scores(qe):
        sims = dense_all @ qe
        best = {}
        for s, d in zip(sims, all_chunk_docs):
            if d not in best or s > best[d]:
                best[d] = float(s)
        return best

    def nn_boost_scores(q: str, ban_doc=None):
        """Soft boost from highly similar train questions' gold docs."""
        sims = cosine_similarity(q_tf.transform([q]), Qmat)[0]
        boost = defaultdict(float)
        for j in np.argsort(-sims)[:12]:
            if str(train.iloc[j].question) == q:
                continue
            if ban_doc is not None and train_gold[j] == ban_doc:
                continue
            s = float(sims[j])
            if s < 0.52:
                break
            boost[train_gold[j]] += s
        return boost

    def first_stage(q: str, ban_doc=None):
        base = tok(q)
        bm1, sd, best = bm25_scores(base)
        bger, bgem, bgef = dense_scores(bge_q[q], bge_chunk, bge_full, bge_docs)
        ftr, ftm, ftf = dense_scores(ft_q[q], ft_chunk, ft_full, ft_docs)
        dall = dense_all_scores(ft_q[q])
        dall_rank = [d for d, _ in sorted(dall.items(), key=lambda x: -x[1])[:220]]
        sw = cosine_similarity(tfw.transform([" ".join(base)]), Dw)[0]
        sc = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:220]],
                [doc_ids[i] for i in np.argsort(sc)[::-1][:220]],
            ]
        )
        nn = nn_boost_scores(q, ban_doc=ban_doc)
        nn_rank = [d for d, _ in sorted(nn.items(), key=lambda x: -x[1])]
        lists = [bm1, ftr, dall_rank, bger, tf]
        weights = [1.45, 1.35, 1.4, 1.05, 0.85]
        if nn_rank:
            lists.append(nn_rank)
            weights.append(0.9)
        fused = rrf(lists, weights=weights)[:CAND]
        return {
            "cands": fused,
            "sd": sd,
            "best": best,
            "bgem": bgem,
            "bgef": bgef,
            "ftm": ftm,
            "ftf": ftf,
            "dall": dall,
            "sw": sw,
            "sc": sc,
            "bm1": bm1,
            "ftr": ftr,
            "bger": bger,
            "nn": nn,
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
            long_hits = sum(1 for t in qt if t in dset and len(t) >= 7)
            X.append(
                [
                    float(pack["sd"][i]),
                    float(pack["best"].get(d, 0)),
                    1.0 / (rank + 1),
                    float(pack["sw"][i]),
                    float(pack["sc"][i]),
                    float(pack["bgem"].get(d, 0)),
                    float(pack["bgef"][i]),
                    float(pack["ftm"].get(d, 0)),
                    float(pack["ftf"][i]),
                    float(pack["dall"].get(d, 0)),
                    float(inter),
                    float(inter / union),
                    float(uniq),
                    float(long_hits),
                    float(lens[i]),
                    float(bm_rank.get(d, 400)),
                    float(ft_rank.get(d, 400)),
                    float(bge_rank.get(d, 400)),
                    float(pack["nn"].get(d, 0)),
                ]
            )
        return np.asarray(X, np.float32)

    print("features...")
    Xs, ys, gs = [], [], []
    doc_group = {d: i for i, d in enumerate(sorted(set(train.gold_doc_id)))}
    q_groups = []
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        # ban same gold doc in NN during train feature building for less leakage
        pack = first_stage(q, ban_doc=g)
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

    print("CV by gold_doc + honest eval...")
    gkf = GroupKFold(5)
    qidx = np.arange(len(train))
    recalls_h, iters = [], []
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
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbosity=-1,
                label_gain=list(range(2)),
            ),
            dtr,
            num_boost_round=400,
            valid_sets=[dve],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        hits_h = 0
        for qi in te_q:
            row = train.iloc[int(qi)]
            q = str(row.question)
            gold = row.gold_doc_id
            pack = first_stage(q, ban_doc=gold)  # honest: no same-doc NN
            scores = booster.predict(make_feats(q, pack))
            ltr = [pack["cands"][i] for i in np.argsort(-scores)]
            final = rrf(
                [ltr, pack["cands"], pack["bm1"][:CAND]], weights=[1.75, 0.8, 0.7]
            )[:TOP_K]
            if gold in final:
                hits_h += 1
        honest = hits_h / len(te_q)
        print(f"fold {fold}: honest={honest:.4f} iter={booster.best_iteration}")
        recalls_h.append(honest)
        iters.append(booster.best_iteration or 120)

    cv_h = float(np.mean(recalls_h))
    n_est = int(np.median(iters))
    print(f"CV honest={cv_h:.4f} rounds={n_est}")

    booster = lgb.train(
        dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5],
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=15,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbosity=-1,
            label_gain=list(range(2)),
        ),
        lgb.Dataset(X, y, group=[CAND] * len(train), feature_name=FEATS),
        num_boost_round=max(n_est, 130),
    )

    def retrieve(q: str):
        pack = first_stage(q, ban_doc=None)  # allow soft NN on test
        scores = booster.predict(make_feats(q, pack))
        ltr = [pack["cands"][i] for i in np.argsort(-scores)]
        return rrf([ltr, pack["cands"], pack["bm1"][:CAND]], weights=[1.75, 0.8, 0.7])[
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
    with open(OUT / "metrics_v6.txt", "w") as f:
        f.write(f"cv_honest_recall@5={cv_h:.6f}\n")
        f.write(f"train_gold_frac={frac:.6f}\n")
        f.write(f"unique_docs={sub.doc_id.nunique()}\n")


if __name__ == "__main__":
    main()
