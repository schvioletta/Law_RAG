"""Nested ruT5-small: question -> short retrieval keywords from ideal/evidence.

Targets are distinctive lemmas present in the gold document (oracle gap terms).
Evaluates BM25(q+gen) under GroupKFold by gold_doc_id.
"""

from __future__ import annotations

import json
from collections import Counter
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
LEGAL_STOP = STOPWORDS | {
    "суд", "судья", "истец", "ответчик", "дело", "г", "года", "руб", "рубль",
    "адрес", "фио", "представитель", "заседание", "протокол", "москва",
    "мещанский", "районный", "также", "данный", "настоящий", "указанный",
}


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


class KwDataset(Dataset):
    def __init__(self, questions, targets, tokenizer, max_src=128, max_tgt=64):
        self.questions = questions
        self.targets = targets
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        src = self.tok(
            "ключевые: " + self.questions[i],
            max_length=self.max_src,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tgt = self.tok(
            self.targets[i],
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


def make_target(question, ideal, evidence, gold_text, df_counter, n=12) -> str:
    qt = set(tok(question))
    gt = set(tok(gold_text))
    scored = []
    for t in set(tok(ideal) + tok(evidence[:700])):
        if t in qt or len(t) < 5:
            continue
        if t not in gt:
            continue
        df = df_counter[t]
        if df == 0 or df > 160:
            continue
        scored.append((df, -len(t), t))
    scored.sort()
    return " ".join(t for *_, t in scored[:n])


def generate(model, tokenizer, questions, batch_size=8, max_new=48):
    model.eval()
    outs = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(questions), batch_size), desc="gen"):
        batch = questions[i : i + batch_size]
        enc = tokenizer(
            ["ключевые: " + q for q in batch],
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
                length_penalty=0.8,
                early_stopping=True,
            )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


def main():
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    dmap = dict(zip(docs.doc_id, docs.text.astype(str)))
    doc_toks = [tok(t) for t in docs.text]
    bm = BM25Okapi(doc_toks)
    df_counter = Counter()
    for dt in doc_toks:
        for t in set(dt):
            df_counter[t] += 1

    targets = [
        make_target(
            str(r.question),
            str(r.ideal_answer),
            str(r.gold_evidence_text),
            dmap[r.gold_doc_id],
            df_counter,
        )
        for _, r in train.iterrows()
    ]
    print("empty targets", sum(1 for t in targets if not t), "/", len(targets))
    print("ex:", targets[0])

    # oracle
    hit = 0
    for i, r in train.iterrows():
        qt = weighted(tok(str(r.question)) + targets[i].split())
        sc = bm.get_scores(qt)
        hit += r.gold_doc_id in [doc_ids[j] for j in np.argsort(sc)[::-1][:5]]
    print("oracle BM25(q+kw)", hit / len(train))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gkf = GroupKFold(5)
    fold_scores = []
    device = "cpu"

    for fold, (tr, te) in enumerate(gkf.split(train, groups=train.gold_doc_id.values)):
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        tr_ds = KwDataset(
            train.iloc[tr].question.astype(str).tolist(),
            [targets[i] for i in tr],
            tokenizer,
        )
        args = Seq2SeqTrainingArguments(
            output_dir=str(OUT / "rut5_kw" / f"fold{fold}"),
            per_device_train_batch_size=8,
            num_train_epochs=4,
            learning_rate=5e-4,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
            predict_with_generate=False,
            fp16=False,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tr_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer,
        )
        trainer.train()
        gens = generate(model, tokenizer, train.iloc[te].question.astype(str).tolist())
        hit = 0
        for j, idx in enumerate(te):
            r = train.iloc[idx]
            extra = tok(gens[j])
            qt = weighted(tok(str(r.question)) + extra)
            sc = bm.get_scores(qt)
            hit += r.gold_doc_id in [doc_ids[k] for k in np.argsort(sc)[::-1][:5]]
            if j < 3:
                print("Q:", str(r.question)[:100])
                print("GEN:", gens[j])
                print("GOLD_T:", targets[idx])
        score = hit / len(te)
        print(f"fold {fold}: {score:.4f}")
        fold_scores.append(score)
        # free
        del model, trainer
        if fold == 0:
            # only need signal from fold0 first; continue all for completeness
            pass

    print("MEAN", float(np.mean(fold_scores)))
    (OUT / "metrics_kw_hyde.txt").write_text(
        json.dumps({"folds": fold_scores, "mean": float(np.mean(fold_scores))}, indent=2)
    )

    # Train on all data for test expansions (for later fusion experiments)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    tr_ds = KwDataset(train.question.astype(str).tolist(), targets, tokenizer)
    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT / "rut5_kw" / "all"),
        per_device_train_batch_size=8,
        num_train_epochs=4,
        learning_rate=5e-4,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        fp16=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tr_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(OUT / "rut5_kw" / "model")
    tokenizer.save_pretrained(OUT / "rut5_kw" / "model")
    test_gen = generate(model, tokenizer, test.question.astype(str).tolist())
    train_gen = generate(model, tokenizer, train.question.astype(str).tolist())
    pd.DataFrame({"qid": train.qid, "keywords": train_gen}).to_csv(OUT / "train_kw.csv", index=False)
    pd.DataFrame({"qid": test.qid, "keywords": test_gen}).to_csv(OUT / "test_kw.csv", index=False)
    print("saved keywords")


if __name__ == "__main__":
    main()
