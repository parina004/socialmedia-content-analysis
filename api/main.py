"""
api/main.py
FastAPI application — 4 endpoints:
  POST /analyze/synthetic   — synthetic media detection only
  POST /analyze/virality    — virality prediction only
  POST /analyze/both        — both models + LLM explanations
  GET  /health              — liveness check

Run with: uv run uvicorn api.main:app --reload
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

## Add project root to sys.path so model_a, model_b, llm, rag are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import (
    BothResponse,
    HealthResponse,
    SyntheticResponse,
    SyntheticResult,
    ViralityFeatures,
    ViralityResponse,
    ViralityResult,
)
from api.utils import save_upload, schedule_delete, validate_video

## Model + LLM imports
from model_a.predict import predict as model_a_predict
from model_b.predict import predict as model_b_predict
from llm.prompts import format_synthetic_prompt, format_virality_prompt
from llm.llm_client import generate as llm_generate
from rag.retriever import retrieve

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

## Rate limiter — 10 requests per minute per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

app = FastAPI(
    title="Social Media Content Analysis API",
    description="Synthetic media detection + virality prediction with LLM explanations.",
    version="1.0.0",
)

## Rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

## CORS — allow Netlify frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.netlify.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


## Helper: build SyntheticResult from model_a output + LLM explanation
def _run_synthetic(video_path: str) -> SyntheticResult:
    result = model_a_predict(video_path)

    ## Retrieve relevant RAG context for the LLM prompt
    rag_context = retrieve("synthetic media detection deepfake AI-generated")

    prompt = format_synthetic_prompt(
        label=result["label"],
        confidence=result["confidence"],
        efficientnet_score=result["prob_ai"],
        forensic_score=result["prob_deepfake"],
        face_score=result["face_score"],
        rag_context=rag_context,
    )
    explanation = llm_generate(prompt)

    return SyntheticResult(
        label=result["label"],
        confidence=result["confidence"],
        prob_ai=result["prob_ai"],
        prob_deepfake=result["prob_deepfake"],
        face_score=result["face_score"],
        explanation=explanation,
    )


## Helper: build ViralityResult from model_b output + LLM explanation
def _run_virality(
    video_path: str,
    title: str,
    description: str,
    tag_count: int,
    upload_hour: int,
    upload_day: str,
    caption: str,
    hashtags: str,
) -> ViralityResult:
    result = model_b_predict(
        video_path=video_path,
        title=title,
        description=description,
        tag_count=tag_count,
        upload_hour=upload_hour,
        upload_day=upload_day,
    )

    f = result["features"]
    features = ViralityFeatures(
        brisque_score=f["brisque_score"],
        color_vibrancy=f["color_vibrancy"],
        motion_intensity=f["motion_intensity"],
        face_presence_ratio=f["face_presence_ratio"],
        face_emotion_joy=f.get("face_emotion_joy", 0.0),
        face_emotion_surprise=f.get("face_emotion_surprise", 0.0),
        thumbnail_brightness=f.get("thumbnail_brightness", 0.0),
        tempo_bpm=f["tempo_bpm"],
        rms_energy=f["rms_energy"],
        speech_ratio=f["speech_ratio"],
        zero_crossing_rate=f.get("zero_crossing_rate", 0.0),
        beat_strength=f.get("beat_strength", 0.0),
        title_sentiment=f["title_sentiment"],
        title_length=f["title_length"],
        title_has_question=f.get("title_has_question", 0),
        title_has_number=f.get("title_has_number", 0),
        description_length=f.get("description_length", 0),
        tag_count=f["tag_count"],
        upload_hour=f["upload_hour"],
        upload_day=f["upload_day"],
        like_to_view_ratio=f.get("like_to_view_ratio", 0.0),
        comment_to_view_ratio=f.get("comment_to_view_ratio", 0.0),
    )

    ## Retrieve relevant RAG context for the LLM prompt
    rag_context = retrieve("video virality prediction social media engagement")

    prompt = format_virality_prompt(
        virality_label=result["virality_label"],
        viral_probability=result["viral_probability"],
        engagement_percentile=result["engagement_percentile"],
        brisque=f["brisque_score"],
        vibrancy=f["color_vibrancy"],
        motion=f["motion_intensity"],
        face_ratio=f["face_presence_ratio"],
        tempo=f["tempo_bpm"],
        rms_energy=f["rms_energy"],
        speech_ratio=f["speech_ratio"],
        title_sentiment=f["title_sentiment"],
        title_length=f["title_length"],
        tag_count=f["tag_count"],
        upload_hour=f["upload_hour"],
        upload_day=f["upload_day"],
        user_caption=caption,
        user_hashtags=hashtags,
        rag_context=rag_context,
    )
    explanation = llm_generate(prompt)

    return ViralityResult(
        virality_label=result["virality_label"],
        viral_probability=result["viral_probability"],
        engagement_percentile=result["engagement_percentile"],
        features=features,
        explanation=explanation,
    )


## ──────────────────────────────────────────
## Endpoints
## ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", models_loaded=True)


@app.post("/analyze/synthetic", response_model=SyntheticResponse)
@limiter.limit("10/minute")
async def analyze_synthetic(
    request: Request,
    file: UploadFile = File(...),
):
    validate_video(file)
    path = await save_upload(file)
    schedule_delete(path)

    synthetic = _run_synthetic(str(path))
    return SyntheticResponse(synthetic=synthetic)


@app.post("/analyze/virality", response_model=ViralityResponse)
@limiter.limit("10/minute")
async def analyze_virality(
    request: Request,
    file: UploadFile = File(...),
    title: str       = Form(default=""),
    description: str = Form(default=""),
    tag_count: int   = Form(default=5),
    upload_hour: int = Form(default=12),
    upload_day: str  = Form(default="Monday"),
    caption: str     = Form(default=""),
    hashtags: str    = Form(default=""),
):
    validate_video(file)
    path = await save_upload(file)
    schedule_delete(path)

    virality = _run_virality(
        video_path=str(path),
        title=title,
        description=description,
        tag_count=tag_count,
        upload_hour=upload_hour,
        upload_day=upload_day,
        caption=caption,
        hashtags=hashtags,
    )
    return ViralityResponse(virality=virality)


@app.post("/analyze/both", response_model=BothResponse)
@limiter.limit("10/minute")
async def analyze_both(
    request: Request,
    file: UploadFile = File(...),
    title: str       = Form(default=""),
    description: str = Form(default=""),
    tag_count: int   = Form(default=5),
    upload_hour: int = Form(default=12),
    upload_day: str  = Form(default="Monday"),
    caption: str     = Form(default=""),
    hashtags: str    = Form(default=""),
):
    validate_video(file)
    path = await save_upload(file)
    schedule_delete(path)

    video_path = str(path)
    synthetic = _run_synthetic(video_path)
    virality = _run_virality(
        video_path=video_path,
        title=title,
        description=description,
        tag_count=tag_count,
        upload_hour=upload_hour,
        upload_day=upload_day,
        caption=caption,
        hashtags=hashtags,
    )
    return BothResponse(synthetic=synthetic, virality=virality)
