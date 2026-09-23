# Backend

The first FastAPI endpoints serve attention-lab payloads from the Python
transformer code and tiny training runs from the training package.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/attention/example` | Load the default lab sentence |
| `POST /api/attention/run` | Run a submitted sentence with up to 6 tokens and optional causal masking |
| `POST /api/attention/compare` | Return unmasked and causal-masked attention for side-by-side learning |
| `GET /api/attention/task-modes` | List supported learning tasks and their mask rules |
| `GET /api/training/example` | Load the default tiny training configuration |
| `POST /api/training/run` | Run tiny GPT next-token training and return metrics |
| `GET /api/training/latest` | Read the most recent backend training run |

Attention responses include token embeddings, sinusoidal position encodings,
combined attention inputs, Q/K/V projections, masks, weights, and context
vectors. They also include the deterministic projection weight matrices and
attention scale needed by the frontend math inspector.

## Run locally

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The frontend dev server proxies `/api` requests to this backend.
