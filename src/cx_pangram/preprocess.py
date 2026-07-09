"""Input normalization, ported verbatim from pangramlabs/EditLens (scripts/preprocess.py).

EditLens was trained on text passed through `clean_text`; inference inputs must match
or scores drift off-distribution. The lowercasing in particular is load-bearing: the
released checkpoints expect lowercased input despite cased backbones.

`remove_think_tag` splits without a maxsplit, so when `</think>` appears more than
once, text between the first and second occurrence is kept and the remainder dropped.
That is an upstream quirk preserved deliberately: patching it would move inference
off the training distribution's preprocessing. Fidelity beats local correctness here.
"""

import re

import emoji

BOILERPLATE_STARTS = [
    "Sure",
    "Here",
    "Abstract",
    "Title",
    "I'm happy to help",
    "Certainly",
]


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_emoji(text):
    return emoji.demojize(text)


def remove_think_tag(text):
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    return text


def remove_ai_header(text):
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) == 0:
        return text
    first_paragraph = paragraphs[0]
    first_paragraph = re.sub(r"^[^a-zA-Z0-9]*", "", first_paragraph)
    first_paragraph = emoji.replace_emoji(first_paragraph, "")
    if any(first_paragraph.startswith(phrase) for phrase in BOILERPLATE_STARTS):
        if len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return text


def clean_text(text):
    text = normalize_emoji(text)
    text = remove_think_tag(text)
    text = remove_ai_header(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def count_words(text):
    return len(re.findall(r"\b\w+\b", text))
