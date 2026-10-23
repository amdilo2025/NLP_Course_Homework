"""Transparent Chinese character/n-gram tokenization, without a model."""
import re
import unicodedata

MODES = ("char", "char_bigram")
# These are individual function characters, not business terms or negation.
STOP_CHARS = frozenset("的了是在我你您请问吗呢啊呀么这那个们很也就都着及与和或于为以而")
STOP_WORDS = frozenset(("the", "a", "an", "is", "are", "of", "to", "and"))
TOKEN_RUN = re.compile(r"[\u3400-\u9fff]+|[a-z]+(?:[a-z0-9_-]*[a-z0-9])?|\d+(?:\.\d+)?", re.I)
HAN_RUN = re.compile(r"^[\u3400-\u9fff]+$")


def tokenize(text: str, mode: str = "char_bigram") -> list[str]:
    if mode not in MODES:
        raise ValueError(f"Unknown tokenizer: {mode}; choose from {MODES}")
    text = unicodedata.normalize("NFKC", text).lower()
    tokens = []
    for match in TOKEN_RUN.finditer(text):
        value = match.group()
        if HAN_RUN.fullmatch(value):
            tokens.extend(char for char in value if char not in STOP_CHARS)
            if mode == "char_bigram":
                # Build bigrams before removing stop characters: never join across
                # punctuation, an English run, a number, or a removed character.
                tokens.extend(value[i:i + 2] for i in range(len(value) - 1)
                              if all(c not in STOP_CHARS for c in value[i:i + 2]))
        elif value not in STOP_WORDS:
            tokens.append(value)
    return tokens
