"""Fast strong retrieval: BM25 + expansion + BGE-m3 + Russian CE rerank."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

TOP_K = 5
RERANK_N = 20


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


def ce_snip(text: str, max_chars: int = 1600) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    # head + a middle window often holds motivirovka
    mid = len(text) // 3
    return (text[: max_chars // 2] + "\n" + text[mid : mid + max_chars // 2])[:max_chars]


def main():
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    dmap = dict(zip(docs.doc_id, docs.text.astype(str)))

    print("BM25...")
    doc_toks = [tok(t) for t in tqdm(docs.text, desc="docs")]
    bm25d = BM25Okapi(doc_toks)
    chunk_toks, chunk_doc = [], []
    for did, text in tqdm(list(zip(docs.doc_id, docs.text)), desc="chunks"):
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
    train_ev_tokens = [tok(str(t)) for t in tqdm(train.gold_evidence_text, desc="ev")]

    print("BGE-m3...")
    bge = SentenceTransformer("deepvk/USER-bge-m3", device="cpu")
    dens_texts, dens_docs = [], []
    for did, text in zip(docs.doc_id, docs.text):
        chs = chunk_text(str(text), 1000, 150)
        if len(chs) > 8:
            idxs = np.linspace(0, len(chs) - 1, 8).astype(int)
            chs = [chs[i] for i in idxs]
        for ch in chs:
            dens_texts.append(ch)
            dens_docs.append(did)

    bge_cache = CACHE / "bge_m3_pure.npz"
    data = np.load(bge_cache)
    chunk_emb, full_emb = data["chunk_emb"], data["full_emb"]
    print("bge cache hit", chunk_emb.shape, full_emb.shape)

    allq = list(
        dict.fromkeys(
            train.question.astype(str).tolist() + test.question.astype(str).tolist()
        )
    )
    q_cache = CACHE / "bge_m3_queries.npz"
    qd = np.load(q_cache, allow_pickle=True)
    if list(qd["questions"]) == allq:
        qemb = {q: e for q, e in zip(qd["questions"], qd["emb"])}
        print("query cache hit")
    else:
        emb = bge.encode(
            allq, batch_size=16, normalize_embeddings=True, show_progress_bar=True
        )
        qemb = {q: e for q, e in zip(allq, emb)}
        np.savez_compressed(
            q_cache, questions=np.array(allq, dtype=object), emb=np.stack(list(emb))
        )

    print("Russian CE...")
    ce = CrossEncoder("DiTy/cross-encoder-russian-msmarco", device="cpu", max_length=512)

    def expand_query(q: str, n_neighbors=8, add_terms=14) -> list[str]:
        base = tok(q)
        sims = cosine_similarity(q_tf.transform([q]), Qmat)[0]
        bag = Counter()
        used = 0
        for j in np.argsort(-sims):
            if sims[j] < 0.12:
                break
            if str(train.iloc[j].question) == q:
                continue
            for t in train_ev_tokens[j]:
                if len(t) >= 5:
                    bag[t] += 1.0 + float(sims[j])
            used += 1
            if used >= n_neighbors:
                break
        extra = [t for t, _ in bag.most_common(add_terms) if t not in set(base)]
        return base + extra

    def bm25_list(q_tokens, topn=150):
        qw = weighted(q_tokens)
        sd = bm25d.get_scores(qw)
        sc = bm25c.get_scores(qw)
        best = {}
        for i, s in enumerate(sc):
            d = chunk_doc[i]
            if d not in best or s > best[d]:
                best[d] = float(s)
        return rrf(
            [
                [doc_ids[i] for i in np.argsort(sd)[::-1][:topn]],
                [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:topn]],
            ],
            weights=[1.0, 1.15],
        )

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
        return [d for d, _ in sorted(best.items(), key=lambda x: -x[1])[:topn]]

    def first_stage(q: str, topn=50):
        base = tok(q)
        exp = expand_query(q)
        bm1 = bm25_list(base)
        bm2 = bm25_list(exp)
        dens = dense_list(q)
        sw = cosine_similarity(tfw.transform([" ".join(base)]), Dw)[0]
        sc = cosine_similarity(tfc.transform([q]), Dc)[0]
        tf = rrf(
            [
                [doc_ids[i] for i in np.argsort(sw)[::-1][:150]],
                [doc_ids[i] for i in np.argsort(sc)[::-1][:150]],
            ]
        )
        return rrf([bm1, bm2, dens, tf], weights=[1.35, 1.05, 1.25, 0.75])[:topn]

    def ce_rerank(q: str, cands: list[str], topk=TOP_K):
        pairs = [(q, ce_snip(dmap[d])) for d in cands]
        scores = np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False))
        fused = defaultdict(float)
        for r, d in enumerate(cands):
            fused[d] += 0.8 / (60 + r + 1)
        for r, i in enumerate(np.argsort(-scores)):
            fused[cands[i]] += 1.6 / (60 + r + 1)
        return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])[:topk]]

    def retrieve(q: str):
        return ce_rerank(q, first_stage(q, 50)[:RERANK_N], TOP_K)

    # Fast first-stage eval on all train
    print("Eval first-stage on train...")
    ranks_fs = []
    for q, g in tqdm(
        zip(train.question.astype(str), train.gold_doc_id), total=len(train)
    ):
        fs = first_stage(q, 50)
        ranks_fs.append(fs.index(g) + 1 if g in fs else 999)
    m_fs = {f"R@{k}": float(np.mean([r <= k for r in ranks_fs])) for k in (1, 5, 10, 20)}
    print("first_stage", m_fs)

    # CE eval on sample
    print("Eval CE on 150 train sample...")
    sample = train.sample(150, random_state=42)
    ranks_ce = []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        q = str(row.question)
        g = row.gold_doc_id
        ranked = retrieve(q)
        # deeper list for recall curve
        fs = first_stage(q, 50)
        deep = ce_rerank(q, fs[:RERANK_N], topk=20)
        deep += [d for d in fs if d not in set(deep)]
        ranks_ce.append(deep.index(g) + 1 if g in deep else 999)
    m_ce = {f"R@{k}": float(np.mean([r <= k for r in ranks_ce])) for k in (1, 5, 10, 20)}
    print("ce_sample", m_ce)

    print("Predict test...")
    rows = []
    for _, row in tqdm(test.iterrows(), total=len(test), desc="test"):
        for d in retrieve(str(row.question)):
            rows.append({"qid": row.qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(OUT / "submission.csv", index=False)
    sub.to_csv(ROOT / "submission.csv", index=False)
    frac = float(sub.doc_id.isin(set(train.gold_doc_id)).mean())
    print(f"unique_docs={sub.doc_id.nunique()} train_gold_frac={frac:.3f}")
    with open(OUT / "metrics_v2.txt", "w") as f:
        f.write(f"first_stage={m_fs}\n")
        f.write(f"ce_sample={m_ce}\n")
        f.write(f"train_gold_frac={frac:.6f}\n")


if __name__ == "__main__":
    main()
