"""HTTP API for MiniGPT Studio labs."""

from dataclasses import asdict
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from training.train_tiny_gpt import run_training
from transformer_core.attention import (
    TASK_MODES,
    build_attention_lab_example,
    build_i_love_transformers_example,
)

app = FastAPI(title="MiniGPT Studio API")


class AttentionRunRequest(BaseModel):
    """Text submitted to the small attention lab."""

    text: str = Field(min_length=1, max_length=120)
    causal_mask: bool = False
    task_mode: str | None = None


class TrainingRunRequest(BaseModel):
    """Configuration for the tiny GPT training lab."""

    epochs: int = Field(default=40, ge=1, le=200)
    learning_rate: float = Field(default=0.08, gt=0, le=1)
    embedding_learning_rate: float = Field(default=0.01, gt=0, le=1)
    context_size: int = Field(default=1, ge=1, le=5)
    train_embeddings: bool = True
    train_transformer: bool = False


RUNS_DIRECTORY = Path("runs")
LATEST_TRAINING_RUN = RUNS_DIRECTORY / "tiny_gpt_backend_latest.json"


@app.get("/api/attention/example")
def get_attention_example() -> dict[str, object]:
    """Return the deterministic single-head attention lab payload."""
    return build_i_love_transformers_example().to_payload()


@app.post("/api/attention/run")
def run_attention(request: AttentionRunRequest) -> dict[str, object]:
    """Return attention tensors for a short lab sentence."""
    text = request.text.strip()
    tokens = text.split()
    if not tokens:
        raise HTTPException(status_code=422, detail="text must include a token")
    if len(tokens) > 6:
        raise HTTPException(status_code=422, detail="text supports up to 6 tokens")

    return build_attention_lab_example(
        text,
        causal_mask=request.causal_mask,
        task_mode=request.task_mode,
    ).to_payload()


@app.post("/api/attention/compare")
def compare_attention(request: AttentionRunRequest) -> dict[str, object]:
    """Return unmasked and causal-masked attention for one lab sentence."""
    text = request.text.strip()
    tokens = text.split()
    if not tokens:
        raise HTTPException(status_code=422, detail="text must include a token")
    if len(tokens) > 6:
        raise HTTPException(status_code=422, detail="text supports up to 6 tokens")

    return {
        "text": text,
        "unmasked": build_attention_lab_example(
            text,
            task_mode="classification",
        ).to_payload(),
        "masked": build_attention_lab_example(text, task_mode="next_token").to_payload(),
    }


@app.get("/api/attention/task-modes")
def get_task_modes() -> dict[str, object]:
    """Return available attention learning task modes."""
    return {"task_modes": TASK_MODES}


@app.post("/api/training/run")
def run_tiny_gpt_training(request: TrainingRunRequest) -> dict[str, object]:
    """Run the tiny GPT training lab and return metrics for visualization."""
    try:
        summary = run_training(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            embedding_learning_rate=request.embedding_learning_rate,
            context_size=request.context_size,
            train_embeddings=request.train_embeddings,
            train_transformer=request.train_transformer,
            output_path=LATEST_TRAINING_RUN,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return _training_payload(asdict(summary), LATEST_TRAINING_RUN)


@app.get("/api/training/latest")
def get_latest_tiny_gpt_training() -> dict[str, object]:
    """Return the most recent backend training run."""
    if not LATEST_TRAINING_RUN.exists():
        raise HTTPException(status_code=404, detail="no training run has been created")

    payload = json.loads(LATEST_TRAINING_RUN.read_text(encoding="utf-8"))
    return _training_payload(payload, LATEST_TRAINING_RUN)


@app.get("/api/training/example")
def get_tiny_gpt_training_example() -> dict[str, object]:
    """Return the default tiny GPT training configuration."""
    return {
        "default_request": TrainingRunRequest().model_dump(),
        "vocabulary": {
            "<unk>": 0,
            "i": 1,
            "love": 2,
            "transformers": 3,
        },
        "corpus_pattern": ["i", "love", "transformers"],
        "task": "next-token prediction",
        "trainable_parameters": [
            "vocabulary projection weights",
            "vocabulary projection biases",
            "token embeddings when train_embeddings is true",
            "decoder blocks when train_transformer is true",
        ],
    }


def _training_payload(
    summary: dict[str, object],
    run_path: Path,
) -> dict[str, object]:
    """Add API metadata around a JSON-friendly training summary."""
    return {
        "run_path": str(run_path),
        "summary": summary,
    }
