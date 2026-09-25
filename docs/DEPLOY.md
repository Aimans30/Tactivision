# Deploying TactiVision (Render + Docker)

This repo ships a **dashboard-first** container for Render.

- Serves the FastAPI API + static web UI
- Seeds a small demo run (`short_sparse`) so the UI is not empty
- **Does not** bake YOLO weights or torch into the image (too large / no GPU on free tier)
- Browser upload / full CV pipeline defaults to **off** in the cloud (`TACTIVISION_UPLOADS_ENABLED=0`)

Full detection/tracking stays a **local** workflow (`pip install -e ".[dev,web,detection]"` + model weights under `models/`).

## Option A — Render Blueprint (recommended)

1. Push this repo to GitHub (already configured as `origin`).
2. In [Render](https://dashboard.render.com/) → **New** → **Blueprint** → select the repo.
3. Render reads `render.yaml` and creates the `tactivision` web service.
4. Wait for the first Docker build, then open the service URL.
5. Health check: `GET /api/health` should return `{"status":"ok",...}`.

Free-tier notes:

- Disk is **ephemeral** — uploads and new processed runs disappear on restart/redeploy.
- Cold starts can take ~30–60s after idle spin-down.
- Do not enable uploads unless you switch to a larger plan, mount a disk, and ship models yourself.

## Option B — Manual Docker web service on Render

1. **New** → **Web Service** → connect the repo.
2. Runtime: **Docker** (Dockerfile at repo root).
3. Health check path: `/api/health`
4. Env vars:
   - `TACTIVISION_UPLOADS_ENABLED=0`
   - `PYTHONUNBUFFERED=1`

Render sets `PORT` automatically; the entrypoint binds `0.0.0.0:$PORT`.

## Local Docker smoke test

```bash
docker build -t tactivision .
docker run --rm -p 8000:8000 tactivision
```

Open http://127.0.0.1:8000 — you should see the seeded `short_sparse` run.

## Enabling uploads on a private/paid instance (optional)

Only if you know what you are doing:

1. Install detection extras and copy weights into the image (or a mounted volume).
2. Set `TACTIVISION_UPLOADS_ENABLED=1`.
3. Prefer a persistent disk for `data/processed` and `data/raw`.
4. Expect slow CPU-only runs; this is still portfolio/demo infrastructure, not production CV SaaS.

## Files

| File | Role |
|------|------|
| `Dockerfile` | Slim Python 3.12 image + web extras + headless OpenCV |
| `.dockerignore` | Keeps weights/videos/local data out of the build context |
| `render.yaml` | Render Blueprint for a free Docker web service |
| `scripts/docker_entrypoint.sh` | Seeds demo run, starts uvicorn |
| `deploy/seed/short_sparse/` | Tiny cached analytics demo (no video) |
