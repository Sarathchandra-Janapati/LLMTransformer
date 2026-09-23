# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniGPT Studio is a staged learning and product project for building small language-model systems from first principles. It demonstrates transformer architecture through a React-based attention visualizer backed by a Python implementation using NumPy.

**Current Status**: Milestone 2 (Attention Lab) — single-head self-attention visualization with attention weights, Q/K/V projections, and causal masking support.

**Key Goal**: Build explainable transformer components layer-by-layer, making each component visible through the web UI before moving to the next stage.

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, NumPy, uvicorn
- **Frontend**: React 19, Vite 7, ES modules
- **Testing**: pytest (Python), vitest (JavaScript, if added)
- **Packaging**: setuptools with pyproject.toml
- **Vectorization**: Pure NumPy (no PyTorch yet)

## Architecture Overview

The project is organized as a monorepo with the following structure:

### Core Packages (Python)

- **`transformer_core/`**: The heart of the project
  - `foundations/`: Manual linear algebra (dot product, matrix multiply) and neural network primitives (Neuron, DenseLayer, TwoLayerRegressionNetwork, activations)
  - `attention/`: Single-head self-attention implementation (SingleHeadSelfAttention), attention lab examples, sinusoidal position encodings, softmax, causal masking
- **`backend/`**: FastAPI HTTP API that exposes attention lab payloads
  - Two main endpoints: GET `/api/attention/example` (default sentence) and POST `/api/attention/run` (user input, up to 6 tokens)
- **`tokenizer/`**: WordTokenizer for simple whitespace-based tokenization with a fixed vocabulary
- **`training/`, `fine_tuning/`, `rag/`, `vector_db/`**: Placeholder packages for future milestones

### Frontend (JavaScript/React)

- **`frontend/src/App.jsx`**: Main React component with:
  - Attention heatmap visualization (color-coded weight matrix)
  - Tensor inspector showing intermediate values (embeddings, Q/K/V, etc.)
  - Interactive text input (up to 6 tokens) with causal masking toggle
  - Formatted value display with worked math for selected cells
- **`frontend/src/data/`**: Supporting utilities for rendering
- **Vite proxy**: `/api` requests forward to backend at `http://127.0.0.1:8000`

### Build and Configuration

- **`pyproject.toml`**: Python dependencies (fastapi, numpy, uvicorn) + dev dependencies (pytest, httpx)
- **`frontend/package.json`**: React, Vite, @vitejs/plugin-react
- **`frontend/vite.config.js`**: Vite setup with React plugin and API proxy

### Testing and Docs

- **`tests/`**: pytest suite covering foundations, neural networks, attention, and backend API
- **`docs/roadmap.md`**: Milestone roadmap from foundation to full transformer
- **`notebooks/`**: Exploratory learning materials (reserved)
- **`datasets/`**: Data folder structure (reserved)

## Development Setup

### Python Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the project and dev dependencies
python -m pip install -e ".[dev]"
```

### Running Tests

```powershell
# Run all tests
pytest

# Run a single test file
pytest tests/test_linear_algebra.py

# Run a specific test
pytest tests/test_attention_lab.py::test_attention_lab_example_uses_one_embedding_per_sentence_token

# Run with verbose output
pytest -v
```

### Running the Backend

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The backend serves two endpoints:
- `GET /api/attention/example`: Returns deterministic "I love transformers" example
- `POST /api/attention/run`: Accepts JSON with `text` (1-6 tokens) and optional `causal_mask` boolean

### Running the Frontend

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The dev server starts at `http://127.0.0.1:5173` by default. Vite proxies API calls to the backend.

## Key Conventions and Patterns

### Code Organization

- **Pure NumPy implementations**: All transformer math uses NumPy arrays without PyTorch shortcuts. This keeps implementations visible and educational.
- **Dataclasses for structure**: Use `@dataclass` for immutable parameter bundles and results (see `AttentionResult`, `Neuron`, `SingleHeadSelfAttention`).
- **Validation in methods**: Input validation happens in `_validate_inputs()` or at method entry (see `SingleHeadSelfAttention.forward()`).
- **Frozen dataclasses for outputs**: Results are frozen (`@dataclass(frozen=True)`) to prevent accidental modification.

### Tensor Shapes

All operations use NumPy array semantics:
- Embeddings and attention inputs: `(token_count, embedding_dim)`
- Weight matrices: `(input_dim, output_dim)`
- Attention masks: `(token_count, token_count)` boolean arrays
- Scores and weights: `(token_count, token_count)`

### Position Encodings

Sinusoidal position encodings are computed independently and added to embeddings:
- Dimensions 0, 2, 4, ... use sine
- Dimensions 1, 3, 5, ... use cosine
- Scale = 10000^(dimension / width)

### Attention Lab Examples

The `lab.py` module provides a deterministic small example with:
- Fixed vocabulary of 7 tokens
- 3D embeddings (hand-crafted, not learned)
- Deterministic Q/K/V projection matrices
- Output as a JSON-serializable `AttentionLabExample` with a `.to_payload()` method for frontend consumption

### Backend API Design

The FastAPI app is minimal and focused:
- Request validation in Pydantic models (text length, token limits)
- Raises `HTTPException` for validation errors
- Returns tensor data as nested lists (JSON-serializable NumPy arrays)
- Endpoints are stateless; no training state or sessions

### Frontend Rendering

The React component uses:
- Controlled input for text and causal masking toggle
- CSS Grid for the heatmap (with CSS custom properties for row/column counts)
- Interactive buttons for selecting query tokens and attention cells
- Formatted output for numerical values (fixed to 3 decimal places)
- Null representation for masked cells in attention scores

## Important Implementation Details

### Single-Head Self-Attention Forward Pass

1. Project embeddings into Q, K, V using weight matrices
2. Compute scaled dot-product scores: (Q @ K.T) / sqrt(d_k)
3. Apply causal mask if enabled (set future positions to -inf)
4. Apply softmax row-wise to get weights
5. Weight values: context = weights @ V
6. Return `AttentionResult` with all intermediates for inspection

### Causal Masking

The causal mask is an upper-triangular boolean matrix (excluding diagonal):
- `mask[i, j] = True` if j > i (future position, should be masked)
- Applied by setting scores to -infinity before softmax
- Weights for masked positions become ~0 after softmax

### Two-Layer Regression Network (Foundations)

Used to demonstrate forward and backward passes:
- Input → Dense Layer + ReLU → Hidden Layer → Dense Layer + MSE Loss
- `train_step()` computes explicit backpropagation with gradient descent
- Useful for understanding how neural network math works

## Testing Notes

- All tests are in `tests/` and use pytest with NumPy's testing utilities
- Backend tests use FastAPI's `TestClient` for integration testing
- Test names are descriptive (e.g., `test_softmax_normalizes_each_token_row`)
- Tests include both happy paths and error cases (shape mismatches, invalid inputs)
- Use `pytest.approx()` for floating-point comparisons

## Roadmap Context

Future milestones depend on the current Attention Lab foundation:

- **Milestone 3**: Multi-head attention, layer norm, feed-forward, residual connections, full transformer blocks
- **Later**: Training dashboard, fine-tuning with LoRA, RAG, services, deployment

Each milestone should be runnable, explainable, and leave a UI for the next stage.

## Common Pitfalls

- **Dimension mismatches**: Pay careful attention to tensor shapes; errors often stem from (batch, seq, dim) vs (seq, batch, dim) confusion
- **Attention scale**: Don't forget the sqrt(d_k) scaling in attention scores
- **Causal mask direction**: Upper triangular (j > i) marks future positions that should be masked, not past positions
- **Position encoding scale**: The exponential scale is 10000^(d / width), not 10000^(d / (2*width))
- **Softmax stability**: The implementation shifts values before exponentiation to avoid overflow

## File Reading Strategy for Future Work

When modifying or extending:
1. Check `transformer_core/foundations/` for low-level math primitives
2. Review `transformer_core/attention/single_head.py` for the core attention mechanism
3. Use `transformer_core/attention/lab.py` as the template for new examples
4. Verify backend API endpoints in `backend/app/main.py` match frontend expectations
5. Test with `tests/` before and after changes
6. Frontend changes require understanding the `tensorViews` config in `App.jsx`
