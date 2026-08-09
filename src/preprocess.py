"""Text preprocessing for Russian legal retrieval."""

from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3
from razdel import tokenize

MORPH = pymorphy3.MorphAnalyzer()

STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
    "нельзя", "такой", "им", "более", "всегда", "конечно", "свою", "между",
}


@lru_cache(maxsize=200_000)
def lemmatize(word: str) -> str:
    return MORPH.parse(word)[0].normal_form


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9\s]", " ", text)


def tokenize_lemmas(text: str) -> list[str]:
    text = normalize_text(text)
    tokens: list[str] = []
    for tok in tokenize(text):
        w = tok.text
        if len(w) < 2 or w in STOPWORDS or w.isdigit():
            continue
        lemma = lemmatize(w)
        if lemma not in STOPWORDS and len(lemma) > 1:
            tokens.append(lemma)
    return tokens


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    text = str(text)
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
