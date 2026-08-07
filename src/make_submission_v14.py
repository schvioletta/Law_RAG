"""v14: USER-base (correct query/passage prompts) + e5_evid + BM25/TFIDF + LTR.

Zero-shot USER dense is complementary to FT e5_evid. Document embeddings MUST use
prompt_name='passage' (default is 'query' and destroys retrieval).

Produces:
  outputs/submission_v14.csv
  outputs/submission_v14_fuse_v6.csv
  outputs/metrics_v14.txt

Root submission.csv is replaced only with --replace-submit.
"""

from __future__ import annotations

import argparse
import json
import time
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

TOP_K = 5
CAND = 80
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
FEATS = [
    "bm25_doc",
    "bm25_chunk",
    "rrf_inv",
    "tfidf_word",
    "tfidf_char",
    "user_chunk",
    "user_full",
    "e5_chunk",
    "e5_full",
    "overlap",
    "jaccard",
    "uniq_hits",
    "long_hits",
    "doc_len",
    "rank_bm",
    "rank_user",
    "rank_e5",
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


def dense_doc_maps(chunk_sims: np.ndarray, chunk_doc: list[str], full_sims: np.ndarray, doc_ids):
    """Aggregate chunk/full cosine sims -> per-doc best score + ranked list."""
    best: dict[str, float] = {}
    for s, d in zip(chunk_sims, chunk_doc):
        s = float(s)
        if d not in best or s > best[d]:
            best[d] = s
    for i, s in enumerate(full_sims):
        d = doc_ids[i]
        best[d] = max(best.get(d, -1e9), float(s))
    ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1])]
    return ranked, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace-submit", action="store_true")
    ap.add_argument("--fuse-only-replace", action="store_true", help="replace root with fuse(v6,v14)")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
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
        for ch in chunk_text(str(text), CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_toks.append(tok(ch))
            chunk_doc.append(did)
            chunk_texts.append(ch)
    bm25c = BM25Okapi(chunk_toks)
    print("chunks", len(chunk_texts))

    tfw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120_000)
    Dw = tfw.fit_transform([" ".join(t) for t in doc_toks])
    tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6), min_df=3, max_features=150_000)
    Dc = tfc.fit_transform(docs.text.fillna(""))
    lens = np.array([len(str(t)) for t in docs.text], float)
    lens = (lens - lens.mean()) / (lens.std() + 1e-6)

    print("load e5_evid...")
    e5 = SentenceTransformer(str(OUT / "e5_evid_v10"), device="cpu")
    ez = np.load(CACHE / "e5_evid_v10.npz")
    assert len(ez["chunk_emb"]) == len(chunk_texts), (len(ez["chunk_emb"]), len(chunk_texts))
    e5_chunk, e5_full = ez["chunk_emb"].astype(np.float32), ez["full_emb"].astype(np.float32)

    print("load USER-base + encode with passage/query prompts...")
    user = SentenceTransformer("deepvk/USER-base", device="cpu")
    # Ensure prompts exist even if hub config differs
    user.prompts = {**getattr(user, "prompts", {}), "query": "query: ", "passage": "passage: ", "document": ""}
    ucache = CACHE / "user_base_passage.npz"
    if ucache.exists() and len(np.load(ucache)["chunk_emb"]) == len(chunk_texts):
        uz = np.load(ucache)
        user_chunk, user_full = uz["chunk_emb"].astype(np.float32), uz["full_emb"].astype(np.float32)
        print("user cache hit", user_chunk.shape)
    else:
        t0 = time.time()
        user_chunk = user.encode(
            chunk_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            prompt_name="passage",
        )
        user_full = user.encode(
            [str(t)[:4000] for t in docs.text],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
            prompt_name="passage",
        )
        np.savez_compressed(ucache, chunk_emb=user_chunk, full_emb=user_full)
        print("user encoded", time.time() - t0, user_chunk.shape)

    allq = train.question.astype(str).tolist() + test.question.astype(str).tolist()
    print("encode queries...")
    e5_qe = e5.encode(
        [f"query: {q}" for q in allq],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    user_qe = user.encode(
        allq,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        prompt_name="query",
    ).astype(np.float32)
    train_e5, test_e5 = e5_qe[: len(train)], e5_qe[len(train) :]
    train_u, test_u = user_qe[: len(train)], user_qe[len(train) :]

    print("precompute dense sims...")

    def precompute(qembs, chunk_emb, full_emb):
        out_rank, out_map, out_full = [], [], []
        bs = 48
        for i0 in tqdm(range(0, len(qembs), bs), desc="dense"):
            qb = qembs[i0 : i0 + bs]
            sims = qb @ chunk_emb.T
            fulls = qb @ full_emb.T
            for bi in range(len(qb)):
                ranked, best = dense_doc_maps(sims[bi], chunk_doc, fulls[bi], doc_ids)
                out_rank.append(ranked)
                out_map.append(best)
                out_full.append(fulls[bi])
        return out_rank, out_map, out_full

    tr_u_rank, tr_u_map, tr_u_full = precompute(train_u, user_chunk, user_full)
    te_u_rank, te_u_map, te_u_full = precompute(test_u, user_chunk, user_full)
    tr_e_rank, tr_e_map, tr_e_full = precompute(train_e5, e5_chunk, e5_full)
    te_e_rank, te_e_map, te_e_full = precompute(test_e5, e5_chunk, e5_full)

    def first_stage(q, u_rank, u_map, u_full, e_rank, e_map, e_full, n=CAND):
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
        sw = cosine_similarity(tfw.transform([" ".join(tok(q))]), Dw)[0]
        sch = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf_rank = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:220]],
                [doc_ids[i] for i in np.argsort(sch)[::-1][:220]],
            ]
        )
        fused = rrf(
            [bm_doc, bm_ch, u_rank[:220], e_rank[:220], tf_rank],
            weights=[1.3, 1.4, 1.6, 1.35, 0.8],
        )[:n]
        return {
            "cands": fused,
            "sd": sd,
            "best_ch": best_ch,
            "u_map": u_map,
            "u_full": u_full,
            "e_map": e_map,
            "e_full": e_full,
            "sw": sw,
            "sc": sch,
            "bm_doc": bm_doc,
            "u_rank": u_rank,
            "e_rank": e_rank,
        }

    hits = defaultdict(int)
    for i, row in tqdm(train.iterrows(), total=len(train), desc="fs"):
        pack = first_stage(
            str(row.question),
            tr_u_rank[i],
            tr_u_map[i],
            tr_u_full[i],
            tr_e_rank[i],
            tr_e_map[i],
            tr_e_full[i],
            n=50,
        )
        for k in (1, 5, 10, 20, 50):
            hits[k] += row.gold_doc_id in pack["cands"][:k]
    print("FS", {k: round(hits[k] / len(train), 4) for k in sorted(hits)})

    def make_feats(q, pack):
        qt = set(tok(q))
        bm_rank = {d: r for r, d in enumerate(pack["bm_doc"])}
        u_rank = {d: r for r, d in enumerate(pack["u_rank"])}
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
                    float(pack["u_map"].get(d, 0)),
                    float(pack["u_full"][i]),
                    float(pack["e_map"].get(d, 0)),
                    float(pack["e_full"][i]),
                    float(inter),
                    float(inter / union),
                    float(sum(1 for t in qt if t in dset and len(t) >= 5)),
                    float(sum(1 for t in qt if t in dset and len(t) >= 7)),
                    float(lens[i]),
                    float(bm_rank.get(d, 400)),
                    float(u_rank.get(d, 400)),
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
        pack = first_stage(
            q,
            tr_u_rank[qi],
            tr_u_map[qi],
            tr_u_full[qi],
            tr_e_rank[qi],
            tr_e_map[qi],
            tr_e_full[qi],
        )
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
            q = str(row.question)
            pack = first_stage(
                q,
                tr_u_rank[int(qi)],
                tr_u_map[int(qi)],
                tr_u_full[int(qi)],
                tr_e_rank[int(qi)],
                tr_e_map[int(qi)],
                tr_e_full[int(qi)],
            )
            scores = booster.predict(make_feats(q, pack))
            ltr = [pack["cands"][i] for i in np.argsort(-scores)]
            final = rrf(
                [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["u_rank"][:CAND], pack["e_rank"][:CAND]],
                weights=[1.85, 0.7, 0.75, 1.25, 1.0],
            )[:TOP_K]
            hit += row.gold_doc_id in final
        score = hit / len(te_q)
        print(f"fold {fold}: {score:.4f} iter={booster.best_iteration}")
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

    rows, v14 = [], {}
    for i, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        q = str(row.question)
        pack = first_stage(
            q,
            te_u_rank[i],
            te_u_map[i],
            te_u_full[i],
            te_e_rank[i],
            te_e_map[i],
            te_e_full[i],
        )
        scores = booster.predict(make_feats(q, pack))
        ltr = [pack["cands"][j] for j in np.argsort(-scores)]
        final = rrf(
            [ltr, pack["cands"], pack["bm_doc"][:CAND], pack["u_rank"][:CAND], pack["e_rank"][:CAND]],
            weights=[1.85, 0.7, 0.75, 1.25, 1.0],
        )[:TOP_K]
        v14[row.qid] = final
        for d in final:
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission_v14.csv", index=False)

    v6 = pd.read_csv(ROOT / "submission.csv")
    v6l = {qid: g.doc_id.tolist() for qid, g in v6.groupby("qid", sort=False)}
    fuse_rows, nd = [], 0
    for qid, lst in v14.items():
        # Prefer v14 slightly when USER+e5 disagree with v6; keep v6 anchor
        fused = rrf([v6l[qid], lst], weights=[1.0, 1.2])[:TOP_K]
        if fused != v6l[qid]:
            nd += 1
        for d in fused:
            fuse_rows.append({"qid": qid, "doc_id": d})
    fuse = pd.DataFrame(fuse_rows)
    fuse.to_csv(OUT / "submission_v14_fuse_v6.csv", index=False)

    train_gold = set(train.gold_doc_id)
    metrics = dict(
        cv=cv,
        folds=recalls,
        fs={str(k): hits[k] / len(train) for k in sorted(hits)},
        unique_v14=int(sub.doc_id.nunique()),
        train_gold_frac_v14=float(sub.doc_id.isin(train_gold).mean()),
        unique_fuse=int(fuse.doc_id.nunique()),
        train_gold_frac_fuse=float(fuse.doc_id.isin(train_gold).mean()),
        fuse_diff_queries=nd,
        note="USER-base passage/query prompts + e5_evid + BM25 + LTR; e5_evid FT is full-data (partly leaky CV)",
    )
    print(metrics)
    (OUT / "metrics_v14.txt").write_text(json.dumps(metrics, indent=2))

    if args.replace_submit:
        sub.to_csv(ROOT / "submission.csv", index=False)
        sub.to_csv(OUT / "submission.csv", index=False)
        print("ROOT submission.csv <- raw v14")
    elif args.fuse_only_replace:
        fuse.to_csv(ROOT / "submission.csv", index=False)
        fuse.to_csv(OUT / "submission.csv", index=False)
        print("ROOT submission.csv <- fuse(v6,v14)")
    else:
        print("root submission.csv unchanged (pass --replace-submit or --fuse-only-replace)")


if __name__ == "__main__":
    main()
