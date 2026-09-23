"""Tiny whitespace tokenizer for the first attention lab examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordTokenizer:
    """Map lower-cased whitespace tokens to ids from a fixed vocabulary."""

    vocabulary: dict[str, int]
    unknown_token: str = "<unk>"

    def tokenize(self, text: str) -> list[str]:
        """Split lower-cased text into simple word tokens."""
        return text.lower().split()

    def encode(self, text: str) -> list[int]:
        """Return token ids, using the unknown token id when needed."""
        if self.unknown_token not in self.vocabulary:
            raise ValueError("vocabulary must include the unknown token")

        unknown_id = self.vocabulary[self.unknown_token]
        return [self.vocabulary.get(token, unknown_id) for token in self.tokenize(text)]

