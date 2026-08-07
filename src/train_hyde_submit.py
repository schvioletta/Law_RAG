"""Fast HyDE: fine-tune ruT5-small question→ideal_answer, retrieve with BM25(q+gen).

Oracle BM25(q+true_ideal) ≈ 0.766 Recall@5 — leader territory.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from preprocess import STOPWORDS, tokenize_lemmas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "cointegrated/rut5-small"
SEED = 42
TOP_K = 5
LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "дело", "г", "года", "руб", "рубль",
    "адрес", "фио", "представитель", "заседание", "протокол", "москва",
    "мещанский", "районный", "также", "данный", "настоящий", "указанный",
}


def tok(text: str) -> list[str]:
    return [t for t in tokenize_lemmas(text) if t not in LEGAL_STOP]


class Q2ADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_src=128, max_tgt=192):
        self.questions = questions
        self.answers = answers
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        src = self.tok(
            "ответить: " + self.questions[i],
            max_length=self.max_src,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tgt = self.tok(
            self.answers[i],
            max_length=self.max_tgt,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": tgt["input_ids"],
        }


def generate_answers(model, tokenizer, questions, batch_size=8, max_new=160):
    model.eval()
    outs = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(questions), batch_size), desc="generate"):
        batch = questions[i : i + batch_size]
        enc = tokenizer(
            ["ответить: " + q for q in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


def bm25_recall(queries, golds, bm25, doc_ids, k=5):
    hits = 0
    for q, g in zip(queries, golds):
        scores = bm25.get_scores(tok(q))
        ranked = [doc_ids[j] for j in np.argsort(scores)[::-1][:k]]
        hits += int(g in ranked)
    return hits / len(queries)


def train_model(train_df, out_dir, epochs=4, lr=3e-4):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    ds = Q2ADataset(
        train_df.question.astype(str).tolist(),
        train_df.ideal_answer.astype(str).tolist(),
        tokenizer,
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=8,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
        report_to=[],
        seed=SEED,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    return model, tokenizer


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    bm25 = BM25Okapi([tok(t) for t in tqdm(docs.text, desc="tok docs")])

    print(
        "oracle BM25(q+ideal):",
        bm25_recall(
            (train.question + " " + train.ideal_answer).astype(str),
            train.gold_doc_id,
            bm25,
            doc_ids,
            5,
        ),
    )
    print(
        "BM25(q):",
        bm25_recall(train.question.astype(str), train.gold_doc_id, bm25, doc_ids, 5),
    )

    # 3-fold honest CV (by gold_doc)
    groups = train.gold_doc_id.values
    gkf = GroupKFold(3)
    fold_scores = []
    qidx = np.arange(len(train))
    for fold, (tr, te) in enumerate(gkf.split(qidx, groups=groups)):
        print(f"\n=== HyDE fold {fold} ===")
        model, tokenizer = train_model(
            train.iloc[tr], OUT / f"rut5_fold{fold}", epochs=5, lr=3e-4
        )
        gens = generate_answers(
            model, tokenizer, train.iloc[te].question.astype(str).tolist()
        )
        # save samples
        for j in range(min(3, len(te))):
            print("Q:", train.iloc[te[j]].question)
            print("GEN:", gens[j][:300])
            print("GOLD_A:", str(train.iloc[te[j]].ideal_answer)[:300])
            print("---")
        fused = [
            str(train.iloc[te[j]].question) + " " + gens[j] for j in range(len(te))
        ]
        only_gen = gens
        r_q = bm25_recall(
            train.iloc[te].question.astype(str),
            train.iloc[te].gold_doc_id,
            bm25,
            doc_ids,
            5,
        )
        r_f = bm25_recall(fused, train.iloc[te].gold_doc_id, bm25, doc_ids, 5)
        r_g = bm25_recall(only_gen, train.iloc[te].gold_doc_id, bm25, doc_ids, 5)
        print(f"fold {fold}: q={r_q:.3f} q+gen={r_f:.3f} gen={r_g:.3f}")
        fold_scores.append(r_f)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    cv = float(np.mean(fold_scores))
    print(f"CV HyDE BM25(q+gen) R@5={cv:.4f}")

    # Final model on all train
    model, tokenizer = train_model(train, OUT / "rut5_final", epochs=6, lr=3e-4)
    train_gens = generate_answers(
        model, tokenizer, train.question.astype(str).tolist()
    )
    test_gens = generate_answers(model, tokenizer, test.question.astype(str).tolist())
    pd.DataFrame({"qid": train.qid, "gen_answer": train_gens}).to_csv(
        OUT / "train_hyde.csv", index=False
    )
    pd.DataFrame({"qid": test.qid, "gen_answer": test_gens}).to_csv(
        OUT / "test_hyde.csv", index=False
    )

    # Build hybrid submission: RRF of BM25(q), BM25(q+gen), BM25(gen)
    from collections import defaultdict

    def rrf(lists, k=60, weights=None):
        weights = weights or [1.0] * len(lists)
        scores = defaultdict(float)
        for w, lst in zip(weights, lists):
            for r, d in enumerate(lst):
                scores[d] += w / (k + r + 1)
        return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def retrieve(q, gen, n=TOP_K):
        lists = []
        for text in (q, q + " " + gen, gen):
            sc = bm25.get_scores(tok(text))
            lists.append([doc_ids[j] for j in np.argsort(sc)[::-1][:80]])
        return rrf(lists, weights=[1.0, 1.8, 1.2])[:n]

    # CV-like train score with final gens (optimistic)
    tr_hit = sum(
        int(g in retrieve(str(q), gen))
        for q, gen, g in zip(train.question, train_gens, train.gold_doc_id)
    )
    print(f"train optimistic R@5={tr_hit/len(train):.4f}")

    rows = []
    for qid, q, gen in zip(test.qid, test.question.astype(str), test_gens):
        for d in retrieve(q, gen):
            rows.append({"qid": qid, "doc_id": d})
    sub = pd.DataFrame(rows)
    sub.to_csv(ROOT / "submission.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    frac = float(sub.doc_id.isin(set(train.gold_doc_id)).mean())
    metrics = {
        "cv_hyde_bm25_qgen": cv,
        "fold_scores": fold_scores,
        "train_optimistic": tr_hit / len(train),
        "train_gold_frac": frac,
        "unique_docs": int(sub.doc_id.nunique()),
    }
    print(metrics)
    with open(OUT / "metrics_hyde.txt", "w") as f:
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
