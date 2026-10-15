# Running the App

## Backend (FastAPI)

From the project root (`Stock_Trend_forecaster/`):

```bash
stockenv312\Scripts\activate
uvicorn Backend.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## Frontend (React + Vite)

```bash
cd Frontend
npm run dev
```

Opens at: http://localhost:5173
