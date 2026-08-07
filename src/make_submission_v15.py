"""v15 final: BM25 + nested-style e5 (full FT) + LTR, fused with pinned v6.

Requires outputs/e5_nested_v15/ and outputs/cache/e5_nested_v15.npz from
make_submission_v15_nested.py (or trains briefly if missing).

Replaces root submission.csv with equal RRF(v6, v15) by default
(real set_diff vs LB-0.52 anchor — not a restore of v6).
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
TOP_K, CAND = 5, 80
LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "дело", "г", "года", "руб", "рубль",
    "адрес", "фио", "представитель", "заседание", "протокол", "москва",
    "мещанский", "районный", "также", "данный", "настоящий", "указанный",
}
FEATS = [
    "bm25_doc", "bm25_chunk", "rrf_inv", "tfidf_word", "tfidf_char",
    "e5_chunk", "e5_full", "overlap", "jaccard", "uniq_hits", "long_hits",
    "doc_len", "rank_bm", "rank_e5",
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
    ap.add_argument("--replace-submit", action="store_true", default=True)
    ap.add_argument("--no-replace", action="store_true")
    args = ap.parse_args()
    replace = args.replace_submit and not args.no_replace

    model_dir = OUT / "e5_nested_v15"
    emb_path = CACHE / "e5_nested_v15.npz"
    # Fallback to e5_evid if nested not ready
    if not model_dir.exists() or not emb_path.exists():
        print("nested model missing; fallback to e5_evid_v10")
        model_dir = OUT / "e5_evid_v10"
        emb_path = CACHE / "e5_evid_v10.npz"

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    id2i = {d: i for i, d in enumerate(doc_ids)}

    print("indexes...")
    doc_toks = [tok(t) for t in tqdm(docs.text)]
    bm25d = BM25Okapi(doc_toks)
    chunk_toks, chunk_doc, chunk_texts = [], [], []
    for did, text in zip(docs.doc_id, docs.text):
        for ch in chunk_text(str(text), 800, 120):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
            chunk_texts.append(ch)
    bm25c = BM25Okapi(chunk_toks)
    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
    Dw = tfw.fit_transform([" ".join(t) for t in doc_toks])
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000)
    Dc = tfc.fit_transform(docs.text.fillna(""))
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    ez = np.load(emb_path)
    assert len(ez["chunk_emb"]) == len(chunk_texts), (len(ez["chunk_emb"]), len(chunk_texts))
    e5_chunk = ez["chunk_emb"].astype(np.float32)
    e5_full = ez["full_emb"].astype(np.float32)
    model = SentenceTransformer(str(model_dir), device="cpu")

    allq = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    qe = model.encode(
        [f"query: {q}" for q in allq],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    train_q, test_q = qe[: len(train)], qe[len(train) :]

    def dense_pack(qe):
        best = {}
        for s, d in zip(e5_chunk @ qe, chunk_doc):
            s = float(s)
            if d not in best or s > best[d]:
                best[d] = s
        full = e5_full @ qe
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
        e_rank, e_map, e_full = dense_pack(qe)
        sw = cosine_similarity(tfw.transform([" ".join(tok(q))]), Dw)[0]
        sch = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf_rank = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:220]],
                [doc_ids[i] for i in np.argsort(sch)[::-1][:220]],
            ]
        )
        fused = rrf([bm_doc, bm_ch, e_rank[:220], tf_rank], weights=[1.4, 1.5, 1.45, 0.85])[:n]
        return dict(
            cands=fused, sd=sd, best_ch=best_ch, e_map=e_map, e_full=e_full,
            sw=sw, sc=sch, bm_doc=bm_doc, e_rank=e_rank,
        )

    def make_feats(q, pack):
        qt = set(tok(q))
        bm_rank = {d: r for r, d in enumerate(pack["bm_doc"])}
        e_rank = {d: r for r, d in enumerate(pack["e_rank"])}
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
                    float(pack["e_map"].get(d, 0)),
                    float(pack["e_full"][i]),
                    float(inter),
                    float(inter / union),
                    float(sum(1 for t in qt if t in dset and len(t) >= 5)),
                    float(sum(1 for t in qt if t in dset and len(t) >= 7)),
                    float(lens[i]),
                    float(bm_rank.get(d, 400)),
                    float(e_rank.get(d, 400)),
                ]
            )
        return np.asarray(X, np.float32)

    print("LTR features...")
    Xs, ys, gs, q_groups = [], [], [], []
    doc_group = {d: i for i, d in enumerate(sorted(set(train.gold_doc_id)))}
    for qi, row in tqdm(train.iterrows(), total=len(train)):
        q = str(row.question)
        g = row.gold_doc_id
        pack = first_stage(q, train_q[qi])
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

    print("CV...")
    recalls, iters = [], []
    for fold, (tr_q, te_q) in enumerate(GroupKFold(5).split(np.arange(len(train)), groups=q_groups)):
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
                objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=0.05,
                num_leaves=31, min_data_in_leaf=15, feature_fraction=0.85,
                bagging_fraction=0.85, bagging_freq=1, verbosity=-1, label_gain=list(range(2)),
            ),
            dtr, num_boost_round=400, valid_sets=[dve],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        hit = 0
        for qi in te_q:
            row = train.iloc[int(qi)]
            q = str(row.question)
            pack = first_stage(q, train_q[int(qi)])
            scores = booster.predict(make_feats(q, pack))
            ltr = [pack["cands"][i] for i in np.argsort(-scores)]
            final = rrf(
                [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["e_rank"][:CAND]],
                weights=[1.9, 0.7, 0.9, 1.15],
            )[:TOP_K]
            hit += row.gold_doc_id in final
        score = hit / len(te_q)
        print(f"fold {fold}: {score:.4f}")
        recalls.append(score)
        iters.append(booster.best_iteration or 120)

    cv = float(np.mean(recalls))
    n_est = int(np.median(iters))
    print("CV", cv, "rounds", n_est)
    booster = lgb.train(
        dict(
            objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=0.05,
            num_leaves=31, min_data_in_leaf=15, feature_fraction=0.85,
            bagging_fraction=0.85, bagging_freq=1, verbosity=-1, label_gain=list(range(2)),
        ),
        lgb.Dataset(X, y, group=[CAND] * len(train), feature_name=FEATS),
        num_boost_round=max(n_est, 140),
    )

    # Always fuse against pinned v6 (LB 0.52), never against current submission.csv
    v6_path = OUT / "submission_v6.csv"
    if not v6_path.exists():
        raise FileNotFoundError(
            f"Missing {v6_path}. Restore with: git show ca2f71a:submission.csv > outputs/submission_v6.csv"
        )
    v6 = pd.read_csv(v6_path)
    v6l = {qid: g.doc_id.tolist() for qid, g in v6.groupby("qid", sort=False)}

    rows, v15 = [], {}
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        q = str(row.question)
        pack = first_stage(q, test_q[i])
        scores = booster.predict(make_feats(q, pack))
        ltr = [pack["cands"][j] for j in np.argsort(-scores)]
        final = rrf(
            [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["e_rank"][:CAND]],
            weights=[1.9, 0.7, 0.9, 1.15],
        )[:TOP_K]
        v15[row.qid] = final
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission_v15_raw.csv", index=False)

    def fuse_with(weights, path, set_key=True):
        fuse_rows, set_diff, list_diff = [], 0, 0
        for qid, lst in v15.items():
            fused = rrf([v6l[qid], lst], weights=weights)[:TOP_K]
            if fused != v6l[qid]:
                list_diff += 1
            if set(fused) != set(v6l[qid][:TOP_K]):
                set_diff += 1
            for d in fused:
                fuse_rows.append({"qid": qid, "doc_id": d})
        fuse = pd.DataFrame(fuse_rows)
        fuse.to_csv(path, index=False)
        return fuse, set_diff, list_diff

    # Mild: slight v6 anchor (often set_diff≈0 on top-5-only RRF — weak for Recall@5)
    mild, mild_sd, mild_ld = fuse_with([1.15, 1.0], OUT / "submission_v15.csv")
    # Equal: real set changes, healthy train_gold_frac — default submit
    equal, eq_sd, eq_ld = fuse_with([1.0, 1.0], OUT / "submission_v15_ltr_equal.csv")
    # Aggressive: prefer v15
    agg, agg_sd, agg_ld = fuse_with([1.0, 1.25], OUT / "submission_v15_aggressive.csv")

    train_gold = set(train.gold_doc_id)
    metrics = dict(
        cv=cv,
        folds=recalls,
        unique_raw=int(sub.doc_id.nunique()),
        train_gold_frac_raw=float(sub.doc_id.isin(train_gold).mean()),
        unique_equal=int(equal.doc_id.nunique()),
        train_gold_frac_equal=float(equal.doc_id.isin(train_gold).mean()),
        equal_set_diff=eq_sd,
        equal_list_diff=eq_ld,
        mild_set_diff=mild_sd,
        agg_set_diff=agg_sd,
        model=str(model_dir),
        note="v15 LTR + equal RRF(v6,v15); NOT a restore of LB 0.52",
    )
    print(metrics)
    (OUT / "metrics_v15.txt").write_text(json.dumps(metrics, indent=2))

    if replace:
        # Equal fuse: set_diff≫0 so Recall@5 can move above v6; tg stays ~0.59
        equal.to_csv(ROOT / "submission.csv", index=False)
        equal.to_csv(OUT / "submission.csv", index=False)
        print(f"ROOT submission.csv <- equal fuse(v6,v15) set_diff={eq_sd}")


if __name__ == "__main__":
    main()
