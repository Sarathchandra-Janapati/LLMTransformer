# Frontend

The first MiniGPT Studio product slice is a React attention visualizer backed by
the FastAPI attention endpoint.

## Run locally

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Start the backend from the repository root before opening the frontend:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The opening screen loads `"I love transformers"` and can run short edited
sentences through the Python attention lab.
