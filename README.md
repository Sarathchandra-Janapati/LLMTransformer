# MiniGPT Studio

MiniGPT Studio is a staged learning and product project for building small
language-model systems from first principles, then making them visible and
usable through a web interface.

## Build order

1. Establish the repo and local development conventions.
2. Implement core neural-network math with NumPy.
3. Build and visualize single-head self-attention.
4. Add transformer blocks, decoder masking, and a tiny GPT training loop.
5. Add training dashboards, generation controls, fine-tuning, RAG, and
   deployment layers only after the smaller systems work.

The current codebase starts at the first executable slice of Step 2:
`transformer_core.foundations` contains manual linear algebra helpers backed by
NumPy arrays and tests that define their behavior.

## Repository map

| Path | Purpose |
| --- | --- |
| `frontend/` | React attention visualizer and later product UI |
| `backend/` | API surface for visualization, training, and inference |
| `transformer_core/` | NumPy and later PyTorch transformer implementations |
| `tokenizer/` | Tokenization experiments and vocabulary tooling |
| `training/` | Training loops, metrics, and checkpoints |
| `fine_tuning/` | LoRA and later fine-tuning experiments |
| `rag/` | Document ingestion and question-answering pipeline |
| `vector_db/` | Vector-store adapters and local indexes |
| `notebooks/` | Exploratory learning notebooks |
| `datasets/` | Dataset notes and local data folders |
| `docs/` | Architecture notes and milestone plans |
| `docker/` | Container definitions as services arrive |

## Local Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## Train The First Tiny Model

This first training run keeps the transformer stack frozen and trains the final
vocabulary projection plus token embeddings for next-token prediction:

```powershell
python -m training.train_tiny_gpt
```

The run writes metrics and final projection weights to
`runs/tiny_gpt_projection_run.json`.

To compare against the earlier projection-only version:

```powershell
python -m training.train_tiny_gpt --projection-only
```

## First product milestone

The first portfolio-visible milestone is an attention visualizer for a short
sentence. It should show tokens, embeddings, Q/K/V projections, attention
scores, softmax weights, and weighted values before the project attempts a full
decoder-only model.
