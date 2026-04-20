# Frontend — Wind Turbine RUL Dashboard

Vite + React + TypeScript SPA that talks to the FastAPI backend
(`backend/app/main.py`).

## Stack

- React 18 + TypeScript
- TanStack Query (polling)
- Plotly.js (charts)
- Tailwind CSS (styling)

## Run locally

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api → :8000)
```

Backend in a separate shell:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

## Generate API types from OpenAPI (optional)

With the backend running on :8000:

```bash
npm run gen:api      # writes src/api/schema.ts from /openapi.json
```
