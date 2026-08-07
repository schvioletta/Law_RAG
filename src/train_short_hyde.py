"""Short HyDE: ruT5 question -> short ideal_answer snippet; BM25(q+gen).

Uses no_repeat_ngram_size to avoid collapse. Nested GroupKFold.
"""

from __future__ import annotations

import json
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


def short_answer(text: str, max_chars=160) -> str:
    text = " ".join(str(text).split())
    # first sentence-ish
    for sep in ".!?":
        if sep in text[: max_chars + 40]:
            cut = text.find(sep)
            if 40 <= cut <= max_chars + 40:
                return text[: cut + 1]
    return text[:max_chars]


class QADataset(Dataset):
    def __init__(self, qs, ans, tokenizer, max_src=140, max_tgt=64):
        self.qs, self.ans, self.tok = qs, ans, tokenizer
        self.max_src, self.max_tgt = max_src, max_tgt

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, i):
        src = self.tok(
            "кратко: " + self.qs[i],
            max_length=self.max_src,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tgt = self.tok(
            self.ans[i],
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


def generate(model, tokenizer, questions, batch_size=8, max_new=56):
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for i in tqdm(range(0, len(questions), batch_size), desc="gen"):
        batch = questions[i : i + batch_size]
        enc = tokenizer(
            ["кратко: " + q for q in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=140,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=0.9,
                early_stopping=True,
            )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs


def main():
    docs = pd.read_csv(DATA / "documents.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    doc_ids = docs.doc_id.tolist()
    bm = BM25Okapi([tok(t) for t in docs.text])
    targets = [short_answer(a) for a in train.ideal_answer]
    print("ex target:", targets[0])

    hit = sum(
        r.gold_doc_id
        in [
            doc_ids[j]
            for j in np.argsort(
                -bm.get_scores(weighted(tok(str(r.question)) + tok(targets[i])))
            )[:5]
        ]
        for i, r in train.iterrows()
    )
    # fix iteration
    hit = 0
    for i, r in train.iterrows():
        sc = bm.get_scores(weighted(tok(str(r.question)) + tok(targets[i])))
        hit += r.gold_doc_id in [doc_ids[j] for j in np.argsort(sc)[::-1][:5]]
    print("oracle short HyDE", hit / len(train))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    fold_scores = []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(train, groups=train.gold_doc_id.values)):
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        # untie warning silence
        try:
            model.config.tie_word_embeddings = False
        except Exception:
            pass
        ds = QADataset(
            train.iloc[tr].question.astype(str).tolist(),
            [targets[i] for i in tr],
            tokenizer,
        )
        args = Seq2SeqTrainingArguments(
            output_dir=str(OUT / "rut5_short" / f"fold{fold}"),
            per_device_train_batch_size=8,
            num_train_epochs=6,
            learning_rate=3e-4,
            logging_steps=100,
            save_strategy="no",
            report_to=[],
            fp16=False,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer,
        )
        trainer.train()
        gens = generate(model, tokenizer, train.iloc[te].question.astype(str).tolist())
        hit = hit_q = 0
        for j, idx in enumerate(te):
            r = train.iloc[idx]
            q = str(r.question)
            sc_q = bm.get_scores(weighted(tok(q)))
            sc = bm.get_scores(weighted(tok(q) + tok(gens[j])))
            hit_q += r.gold_doc_id in [doc_ids[k] for k in np.argsort(sc_q)[::-1][:5]]
            hit += r.gold_doc_id in [doc_ids[k] for k in np.argsort(sc)[::-1][:5]]
            if j < 2:
                print("Q:", q[:100])
                print("GEN:", gens[j])
                print("TGT:", targets[idx])
        print(f"fold {fold}: bm25={hit_q/len(te):.4f} hyde={hit/len(te):.4f}")
        fold_scores.append(hit / len(te))
        del model, trainer
        if fold >= 1:
            # two folds enough for signal; still finish if fast
            pass

    print("MEAN hyde", float(np.mean(fold_scores)))
    (OUT / "metrics_short_hyde.txt").write_text(
        json.dumps({"folds": fold_scores, "mean": float(np.mean(fold_scores))}, indent=2)
    )

    # full train -> test gens for later fusion if mean > bm25
    if float(np.mean(fold_scores)) >= 0.48:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        try:
            model.config.tie_word_embeddings = False
        except Exception:
            pass
        ds = QADataset(train.question.astype(str).tolist(), targets, tokenizer)
        args = Seq2SeqTrainingArguments(
            output_dir=str(OUT / "rut5_short" / "all"),
            per_device_train_batch_size=8,
            num_train_epochs=6,
            learning_rate=3e-4,
            logging_steps=100,
            save_strategy="no",
            report_to=[],
            fp16=False,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(OUT / "rut5_short" / "model")
        tokenizer.save_pretrained(OUT / "rut5_short" / "model")
        test_gen = generate(model, tokenizer, test.question.astype(str).tolist())
        train_gen = generate(model, tokenizer, train.question.astype(str).tolist())
        pd.DataFrame({"qid": test.qid, "hyde": test_gen}).to_csv(OUT / "test_short_hyde.csv", index=False)
        pd.DataFrame({"qid": train.qid, "hyde": train_gen}).to_csv(OUT / "train_short_hyde.csv", index=False)
        print("saved short hyde")


if __name__ == "__main__":
    main()
