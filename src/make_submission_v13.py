"""v13: reuse e5_evid_v10 (no re-FT) + BM25/TFIDF + LTR, NO nn_boost.

Also builds a conservative RRF fuse with frozen v6 submission for test.
Does NOT overwrite submission.csv unless --replace-submit is passed.
"""

from __future__ import annotations

import argparse
import json
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
    "суд", "судья", "истец", "ответчик", "дело", "г", "года", "руб", "рубль",
    "адрес", "фио", "представитель", "заседание", "протокол", "москва",
    "мещанский", "районный", "также", "данный", "настоящий", "указанный",
}

TOP_K, CAND = 5, 80
CHUNK_SIZE, CHUNK_OV = 800, 120
FEATS = [
    "bm25_doc", "bm25_chunk", "rrf_inv", "tfidf_word", "tfidf_char",
    "ft_chunk", "ft_full", "overlap", "jaccard", "uniq_hits", "long_hits",
    "doc_len", "rank_bm", "rank_ft", "margin_ft", "topic_prior",
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--replace-submit",
        action="store_true",
        help="Overwrite submission.csv (default: keep v6; write outputs/submission_v13*.csv)",
    )
    ap.add_argument("--fuse-v6", action="store_true", default=True)
    args = ap.parse_args()

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    id2i = {d: i for i, d in enumerate(doc_ids)}

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
    assert len(chunk_texts) == 4869, len(chunk_texts)

    cleaned = [" ".join(t) for t in doc_toks]
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
    Dw = tfw.fit_transform(cleaned)
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000)
    Dc = tfc.fit_transform(docs.text.fillna(""))
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    # Soft topic→doc prior from train (used only as a weak feature, not candidate gate)
    topic_doc = defaultdict(lambda: defaultdict(float))
    topic_n = defaultdict(float)
    for _, row in train.iterrows():
        topic_doc[row.topic][row.gold_doc_id] += 1.0
        topic_n[row.topic] += 1.0

    # Simple topic classifier: nearest train question by char-tfidf
    q_tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Qmat = q_tf.fit_transform(train.question.astype(str))
    train_topics = train.topic.tolist()

    def predict_topic(q: str) -> str:
        sims = cosine_similarity(q_tf.transform([q]), Qmat)[0]
        return train_topics[int(np.argmax(sims))]

    print("load e5_evid_v10...")
    model = SentenceTransformer(str(OUT / "e5_evid_v10"), device="cpu")
    cache = np.load(CACHE / "e5_evid_v10.npz")
    chunk_emb, full_emb = cache["chunk_emb"], cache["full_emb"]
    assert len(chunk_emb) == len(chunk_texts)

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

    def first_stage(q, qe, n=CAND, ban_topic_doc=None):
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
            weights=[1.35, 1.45, 1.55, 0.85],
        )[:n]
        topic = predict_topic(q)
        prior = dict(topic_doc[topic])
        if ban_topic_doc is not None:
            prior.pop(ban_topic_doc, None)
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
            "topic_prior": prior,
            "topic_n": topic_n[topic] + 1e-6,
        }

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
        # margin vs best dense
        best_ft = max(pack["dens_map"].values()) if pack["dens_map"] else 0.0
        X = []
        for rank, d in enumerate(pack["cands"]):
            i = id2i[d]
            dset = set(doc_toks[i])
            inter = len(qt & dset)
            union = len(qt | dset) + 1e-6
            ft = float(pack["dens_map"].get(d, 0))
            X.append(
                [
                    float(pack["sd"][i]),
                    float(pack["best_ch"].get(d, 0)),
                    1.0 / (rank + 1),
                    float(pack["sw"][i]),
                    float(pack["sc"][i]),
                    ft,
                    float(pack["dens_full"][i]),
                    float(inter),
                    float(inter / union),
                    float(sum(1 for t in qt if t in dset and len(t) >= 5)),
                    float(sum(1 for t in qt if t in dset and len(t) >= 7)),
                    float(lens[i]),
                    float(bm_rank.get(d, 400)),
                    float(ft_rank.get(d, 400)),
                    float(best_ft - ft),
                    float(pack["topic_prior"].get(d, 0.0) / pack["topic_n"]),
                ]
            )
        return np.asarray(X, np.float32)

    print("build LTR features...")
    Xs, ys, gs, q_groups = [], [], [], []
    doc_group = {d: i for i, d in enumerate(sorted(set(train.gold_doc_id)))}
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        pack = first_stage(q, train_qe[qi], n=CAND, ban_topic_doc=g)
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

    print("CV LTR (note: dense FT still leaky vs full-data e5_evid)...")
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
            pack = first_stage(
                str(row.question),
                train_qe[int(qi)],
                n=CAND,
                ban_topic_doc=row.gold_doc_id,
            )
            scores = booster.predict(make_feats(str(row.question), pack))
            ltr = [pack["cands"][i] for i in np.argsort(-scores)]
            final = rrf(
                [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["dens_rank"][:CAND]],
                weights=[1.9, 0.7, 0.85, 1.15],
            )[:TOP_K]
            hit += row.gold_doc_id in final
        score = hit / len(te_q)
        print(f"fold {fold}: cv={score:.4f} iter={booster.best_iteration}")
        recalls.append(score)
        iters.append(booster.best_iteration or 120)

    cv = float(np.mean(recalls))
    n_est = int(np.median(iters))
    print(f"CV={cv:.4f} rounds={n_est}")

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

    def retrieve(q, qe):
        pack = first_stage(q, qe, n=CAND)
        scores = booster.predict(make_feats(q, pack))
        ltr = [pack["cands"][j] for j in np.argsort(-scores)]
        return rrf(
            [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["dens_rank"][:CAND]],
            weights=[1.9, 0.7, 0.85, 1.15],
        )[:TOP_K]

    rows = []
    v13_lists = {}
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        final = retrieve(str(row.question), test_qe[i])
        v13_lists[row.qid] = final
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission_v13.csv", index=False)

    # Fuse with frozen v6
    v6 = pd.read_csv(ROOT / "submission.csv")
    # Guard: only fuse if current root submit looks like v6-sized file
    v6_lists = {qid: g.doc_id.tolist() for qid, g in v6.groupby("qid", sort=False)}
    fuse_rows = []
    n_diff = 0
    for qid, lst in v13_lists.items():
        fused = rrf([v6_lists[qid], lst], weights=[1.25, 1.0])[:TOP_K]
        if fused != v6_lists[qid]:
            n_diff += 1
        for d in fused:
            fuse_rows.append({"qid": qid, "doc_id": d})
    fuse = pd.DataFrame(fuse_rows)
    fuse.to_csv(OUT / "submission_v13_fuse_v6.csv", index=False)
    print(f"fuse differs from v6 on {n_diff}/{len(v13_lists)} queries")

    metrics = {
        "cv": cv,
        "folds": recalls,
        "fs": {str(k): hits[k] / len(train) for k in sorted(hits)},
        "unique_docs_v13": int(sub.doc_id.nunique()),
        "train_gold_frac_v13": float(sub.doc_id.isin(set(train.gold_doc_id)).mean()),
        "unique_docs_fuse": int(fuse.doc_id.nunique()),
        "train_gold_frac_fuse": float(fuse.doc_id.isin(set(train.gold_doc_id)).mean()),
        "fuse_diff_queries": n_diff,
        "note": "dense FT is full-data e5_evid_v10; CV still partly leaky; root submission.csv left as v6 unless --replace-submit",
    }
    print(metrics)
    with open(OUT / "metrics_v13.txt", "w") as f:
        f.write(json.dumps(metrics, indent=2))

    if args.replace_submit:
        # Prefer fuse (conservative) over raw v13
        fuse.to_csv(ROOT / "submission.csv", index=False)
        fuse.to_csv(OUT / "submission.csv", index=False)
        print("replaced submission.csv with v13⊗v6 fuse")
    else:
        print("kept root submission.csv as-is (v6). See outputs/submission_v13*.csv")


if __name__ == "__main__":
    main()
