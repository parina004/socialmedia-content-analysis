"""
api/schemas.py
Pydantic models for request and response validation across all FastAPI endpoints.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


## Shared synthetic detection result
class SyntheticResult(BaseModel):
    label: str                          ## "Real" | "Deepfake" | "AI-Generated"
    confidence: float                   ## 0–100
    prob_ai: float
    prob_deepfake: float
    face_score: Optional[float]         ## None if no face detected
    explanation: str                    ## LLM-generated prose explanation


## Extracted visual + audio + metadata features returned to the frontend
class ViralityFeatures(BaseModel):
    brisque_score: float
    color_vibrancy: float
    motion_intensity: float
    face_presence_ratio: float
    face_emotion_joy: float
    face_emotion_surprise: float
    thumbnail_brightness: float
    tempo_bpm: float
    rms_energy: float
    speech_ratio: float
    zero_crossing_rate: float
    beat_strength: float
    title_sentiment: float
    title_length: int
    title_has_question: int
    title_has_number: int
    description_length: int
    tag_count: int
    upload_hour: int
    upload_day: str
    like_to_view_ratio: float
    comment_to_view_ratio: float


## Shared virality prediction result
class ViralityResult(BaseModel):
    virality_label: str                 ## "Viral" | "Not Viral"
    viral_probability: float            ## 0–100
    engagement_percentile: float        ## 0–100
    features: ViralityFeatures
    explanation: str                    ## LLM-generated prose explanation


## Response for /analyze/synthetic
class SyntheticResponse(BaseModel):
    synthetic: SyntheticResult


## Response for /analyze/virality
class ViralityResponse(BaseModel):
    virality: ViralityResult


## Response for /analyze/both
class BothResponse(BaseModel):
    synthetic: SyntheticResult
    virality: ViralityResult


## Response for /health
class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: bool
