from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model import SpamPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample_spam.csv"
FRONTEND_PATH = BASE_DIR / "frontend"

app = FastAPI(title="Hybrid Feature Selection Spam Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = SpamPipeline(max_features=5000, k_features=2000)


class PredictRequest(BaseModel):
    messages: List[str]


class TrainRequest(BaseModel):
    epochs: int = 12


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/train")
def train(request: TrainRequest) -> Dict:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Training data not found.")
    data = pd.read_csv(DATA_PATH)
    summary = pipeline.train(
        data["text"].tolist(), data["label"].tolist(), epochs=request.epochs
    )
    return {"message": "Training complete", "summary": summary}


@app.post("/api/predict")
def predict(request: PredictRequest) -> Dict:
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
    try:
        results = pipeline.predict(request.messages)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": results}


if FRONTEND_PATH.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
