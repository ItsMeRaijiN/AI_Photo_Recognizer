# AI Photo Recognizer

Web application that detects AI-generated images. A fine-tuned CNN (EfficientNetV2-S or ConvNeXtV2) classifies uploaded photos as **AI-generated** or **real**, explains its decision with Grad-CAM heatmaps, and enriches every result with a set of pluggable image-forensics metrics (blur, noise, texture, color, edges).

## Features

- Single image or whole-folder analysis — files are uploaded from the browser (drag & drop, file/folder picker, clipboard paste, image URL)
- Grad-CAM attention heatmaps with adjustable overlay
- Pluggable custom metrics — drop a `*.py` file with a `calculate()` function into `backend/custom_metrics/` and reload from the admin panel
- User accounts (JWT) with per-user history, result caching by file hash, and CSV export
- Guest mode with session-local history
- Admin panel: user management, system stats, model upload, DB maintenance
- PyTorch (`.pt`) and ONNX (`.onnx`) model support, CPU/CUDA/MPS

## Tech stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.12, FastAPI, SQLAlchemy 2 (SQLite), Pydantic 2, PyTorch / ONNX Runtime, timm, OpenCV, Pillow |
| Frontend | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4, framer-motion, axios |
| Tests | pytest (101 tests), Jest + Testing Library (128 tests) |

## Repository layout

```
backend/
  core/            # config, database, security (JWT, bcrypt)
  models/          # SQLAlchemy models (User, Analysis)
  routers/         # auth, analysis, admin endpoints
  schemas/         # Pydantic request/response models
  services/        # ML engine, batch processor, metrics loader
  custom_metrics/  # pluggable image-forensics metrics
  ml/              # training & evaluation scripts
  tests/
frontend/
  src/app/         # Next.js App Router entry
  src/components/  # Dashboard, UploadArea, AnalysisResult, AdminPanel, ...
  src/lib/         # API client, error helpers, shared types
  src/__tests__/
```

## Getting started

Requirements: **Python 3.12+**, **Node.js 20+**.

### 1. Backend

```powershell
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r backend/requirements.txt

copy .env.example .env           # then edit: set your own secrets
python -m uvicorn backend.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs · health check: http://localhost:8000/health

### 2. Model weights

Trained weights are **not** included in the repository. Place a checkpoint at:

```
runs/experiment/run_<name>/best_model.pt
```

(optionally with a `results.json` containing `backbone` and `best_threshold`) — it is auto-discovered on startup. Alternatively set an explicit `MODEL_PATH` in `.env`. Without a model the API runs in degraded mode and predictions return an error. Training scripts live in `backend/ml/`.

### 3. Frontend

```powershell
cd frontend
npm install
copy .env.local.example .env.local   # points to http://localhost:8000 by default
npm run dev
```

App: http://localhost:3000

### 4. First admin account (optional)

Registration from the UI creates regular users. Create the initial admin once via `POST /admin/bootstrap` (easiest from `/docs`), passing `username`, `password` and `secret_token` equal to `ADMIN_BOOTSTRAP_TOKEN` from your `.env`.

## Tests

```powershell
# backend (repo root)
python -m pytest

# frontend
cd frontend
npm test
npm run lint
```
