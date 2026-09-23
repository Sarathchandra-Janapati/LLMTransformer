"""Run a tiny next-token training experiment.

This trains the final vocabulary projection and token embeddings. The
transformer stack stays frozen so the first deeper training run stays inspectable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from training import (
    build_next_token_examples,
    cross_entropy_loss,
    train_full_model,
    train_projection_and_embeddings,
    train_output_projection,
)
from transformer_core import DecoderStack, LanguageModelHead, TinyGPTModel
from transformer_core import TokenEmbeddingTable
from transformer_core.attention import MultiHeadSelfAttention
from transformer_core.blocks import FeedForwardNetwork, LayerNorm, TransformerBlock


VOCABULARY = {
    "<unk>": 0,
    "i": 1,
    "love": 2,
    "transformers": 3,
}


@dataclass(frozen=True)
class TrainingRunSummary:
    """JSON-friendly summary for a tiny training run."""

    vocabulary: dict[str, int]
    corpus_token_ids: list[int]
    context_size: int
    epochs: int
    learning_rate: float
    embedding_learning_rate: float
    train_embeddings: bool
    train_transformer: bool
    example_count: int
    initial_loss: float
    final_loss: float
    initial_predictions: list[int]
    final_predictions: list[int]
    target_token_ids: list[int]
    metrics: list[dict[str, float | int]]
    token_embeddings: list[list[float]]
    vocabulary_weights: list[list[float]]
    vocabulary_biases: list[float]


def build_demo_model(seed: int = 7) -> TinyGPTModel:
    """Build a deterministic tiny GPT-shaped model for training experiments."""
    rng = np.random.default_rng(seed)
    model_width = 3
    feed_forward_width = 6
    vocabulary_size = len(VOCABULARY)

    return TinyGPTModel(
        token_embedding_table=TokenEmbeddingTable(
            embeddings=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.2, 0.1],
                    [0.3, 1.0, 0.2],
                    [0.1, 0.4, 1.0],
                ],
                dtype=float,
            )
        ),
        language_model=LanguageModelHead(
            decoder=DecoderStack(
                blocks=[
                    TransformerBlock(
                        attention=MultiHeadSelfAttention(
                            query_weights=[
                                rng.normal(0.0, 0.2, size=(model_width, model_width))
                            ],
                            key_weights=[
                                rng.normal(0.0, 0.2, size=(model_width, model_width))
                            ],
                            value_weights=[
                                rng.normal(0.0, 0.2, size=(model_width, model_width))
                            ],
                            output_weights=rng.normal(
                                0.0,
                                0.2,
                                size=(model_width, model_width),
                            ),
                        ),
                        attention_norm=LayerNorm(
                            gamma=np.ones(model_width),
                            beta=np.zeros(model_width),
                        ),
                        feed_forward=FeedForwardNetwork(
                            input_weights=rng.normal(
                                0.0,
                                0.2,
                                size=(model_width, feed_forward_width),
                            ),
                            input_biases=np.zeros(feed_forward_width),
                            output_weights=rng.normal(
                                0.0,
                                0.2,
                                size=(feed_forward_width, model_width),
                            ),
                            output_biases=np.zeros(model_width),
                        ),
                        feed_forward_norm=LayerNorm(
                            gamma=np.ones(model_width),
                            beta=np.zeros(model_width),
                        ),
                    )
                ]
            ),
            final_norm=LayerNorm(gamma=np.ones(model_width), beta=np.zeros(model_width)),
            vocabulary_weights=np.zeros((model_width, vocabulary_size)),
            vocabulary_biases=np.zeros(vocabulary_size),
        ),
    )


def run_training(
    epochs: int,
    learning_rate: float,
    embedding_learning_rate: float,
    context_size: int,
    train_embeddings: bool,
    train_transformer: bool,
    output_path: Path,
) -> TrainingRunSummary:
    """Train the demo model and write a JSON run summary."""
    model = build_demo_model()
    corpus_token_ids = [1, 2, 3] * 12
    examples = build_next_token_examples(corpus_token_ids, context_size=context_size)
    first_example = examples[0]

    initial_result = model.forward(first_example.input_token_ids, causal_mask=True)
    initial_loss = cross_entropy_loss(
        initial_result.language_model.probabilities,
        first_example.target_token_ids,
    )

    if train_transformer:
        history = train_full_model(
            model=model,
            examples=examples,
            learning_rate=learning_rate,
            epochs=epochs,
        )
    elif train_embeddings:
        history = train_projection_and_embeddings(
            model=model,
            examples=examples,
            projection_learning_rate=learning_rate,
            embedding_learning_rate=embedding_learning_rate,
            epochs=epochs,
        )
    else:
        history = train_output_projection(
            model=model,
            examples=examples,
            learning_rate=learning_rate,
            epochs=epochs,
        )

    final_result = model.forward(first_example.input_token_ids, causal_mask=True)
    final_loss = cross_entropy_loss(
        final_result.language_model.probabilities,
        first_example.target_token_ids,
    )

    summary = TrainingRunSummary(
        vocabulary=VOCABULARY,
        corpus_token_ids=corpus_token_ids,
        context_size=context_size,
        epochs=epochs,
        learning_rate=learning_rate,
        embedding_learning_rate=embedding_learning_rate,
        train_embeddings=train_embeddings,
        train_transformer=train_transformer,
        example_count=len(examples),
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_predictions=initial_result.language_model.predicted_token_ids.tolist(),
        final_predictions=final_result.language_model.predicted_token_ids.tolist(),
        target_token_ids=first_example.target_token_ids.tolist(),
        metrics=[asdict(step.metrics) for step in history],
        token_embeddings=model.token_embedding_table.embeddings.tolist(),
        vocabulary_weights=model.language_model.vocabulary_weights.tolist(),
        vocabulary_biases=model.language_model.vocabulary_biases.tolist(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny GPT projection head.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--embedding-learning-rate", type=float, default=0.01)
    parser.add_argument("--context-size", type=int, default=1)
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Skip finite-difference token-embedding training.",
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Train embeddings, decoder blocks, final norm, and vocabulary projection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/tiny_gpt_projection_run.json"),
    )
    args = parser.parse_args()

    summary = run_training(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        embedding_learning_rate=args.embedding_learning_rate,
        context_size=args.context_size,
        train_embeddings=not args.projection_only,
        train_transformer=args.full_model,
        output_path=args.output,
    )

    print("TinyGPT training complete")
    print(f"examples: {summary.example_count}")
    print(f"epochs: {summary.epochs}")
    print(f"learning_rate: {summary.learning_rate}")
    print(f"embedding_learning_rate: {summary.embedding_learning_rate}")
    print(f"train_embeddings: {summary.train_embeddings}")
    print(f"train_transformer: {summary.train_transformer}")
    print(f"initial_loss: {summary.initial_loss:.4f}")
    print(f"final_loss: {summary.final_loss:.4f}")
    print(f"target_token_ids: {summary.target_token_ids}")
    print(f"initial_predictions: {summary.initial_predictions}")
    print(f"final_predictions: {summary.final_predictions}")
    print(f"saved_run: {args.output}")


if __name__ == "__main__":
    main()
