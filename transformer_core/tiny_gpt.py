"""Token-id wrapper around the tiny decoder language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from transformer_core.attention import sinusoidal_position_encodings
from transformer_core.foundations.linear_algebra import FloatArray
from transformer_core.language_modeling import LanguageModelHead, LanguageModelResult


@dataclass(frozen=True)
class TinyGPTResult:
    """Inspectable tensors from token ids through vocabulary probabilities."""

    token_ids: NDArray[np.integer]
    token_embeddings: FloatArray
    position_encodings: FloatArray
    model_inputs: FloatArray
    language_model: LanguageModelResult


@dataclass
class TokenEmbeddingTable:
    """Lookup table that maps integer token ids to learned token vectors."""

    embeddings: FloatArray

    def forward(self, token_ids: Sequence[int] | NDArray[np.integer]) -> FloatArray:
        """Return one embedding row for each token id."""
        token_id_array = np.asarray(token_ids, dtype=int)
        self._validate_token_ids(token_id_array)
        return self.embeddings[token_id_array]

    def _validate_token_ids(self, token_ids: NDArray[np.integer]) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("token embeddings must be a vocabulary-by-width matrix")
        if token_ids.ndim != 1:
            raise ValueError("token ids must be a one-dimensional sequence")
        if token_ids.size == 0:
            raise ValueError("token ids must include at least one token")
        if np.any(token_ids < 0) or np.any(token_ids >= self.embeddings.shape[0]):
            raise ValueError("token id is outside the embedding vocabulary")


@dataclass
class TinyGPTModel:
    """Minimal GPT-style model: tokens -> embeddings -> positions -> decoder."""

    token_embedding_table: TokenEmbeddingTable
    language_model: LanguageModelHead

    def forward(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
        attention_mask: NDArray[np.bool_] | None = None,
        causal_mask: bool = True,
    ) -> TinyGPTResult:
        """Run a full tiny language-model forward pass from token ids."""
        self._validate_model_shapes()

        token_id_array = np.asarray(token_ids, dtype=int)
        token_embeddings = self.token_embedding_table.forward(token_id_array)
        position_encodings = sinusoidal_position_encodings(
            token_count=token_embeddings.shape[0],
            width=token_embeddings.shape[1],
        )
        model_inputs = token_embeddings + position_encodings
        language_model_result = self.language_model.forward(
            model_inputs,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )

        return TinyGPTResult(
            token_ids=token_id_array,
            token_embeddings=token_embeddings,
            position_encodings=position_encodings,
            model_inputs=model_inputs,
            language_model=language_model_result,
        )

    def _validate_model_shapes(self) -> None:
        embeddings = self.token_embedding_table.embeddings
        vocabulary_weights = self.language_model.vocabulary_weights
        if embeddings.ndim != 2:
            raise ValueError("token embeddings must be a vocabulary-by-width matrix")
        if vocabulary_weights.ndim != 2:
            raise ValueError("vocabulary weights must be a model-width-by-vocabulary matrix")
        if embeddings.shape[1] != vocabulary_weights.shape[0]:
            raise ValueError("token embedding width must match the model width")
        if embeddings.shape[0] != vocabulary_weights.shape[1]:
            raise ValueError("input vocabulary size must match output vocabulary size")
