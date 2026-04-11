# api/main.py
# FastAPI backend — the single entry point for all video analysis requests.
# Exposes 3 endpoints: /analyze/synthetic, /analyze/virality, /health
# Security: MIME validation, 100MB file size limit, rate limiting (10 req/min), CORS.
# Privacy: uploaded files are auto-deleted 5 minutes after processing completes.

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Logging — print all INFO+ logs from our modules to the terminal
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(levelname)s  [%(name)s]  %(message)s",
    handlers= [logging.StreamHandler()],
)

load_dotenv()

# App setup

# slowapi uses the client IP as the rate limit key
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Models load lazily on first request — pre-warming caused segfaults on Apple M4
    # due to a Metal/GL context conflict between MediaPipe and PyTorch safetensors loading.
    yield


app = FastAPI(
    title       = "Social Media Content Analysis API",
    description = "Detects synthetic media and predicts virality for uploaded videos.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Register the rate-limit error handler so 429 is returned (not 500) on breach
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow requests only from the Netlify frontend (and localhost for dev)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Strip trailing slashes from each origin to avoid mismatch
_origins = (
    ["*"] if ALLOWED_ORIGINS == "*"
    else [o.strip().rstrip("/") for o in ALLOWED_ORIGINS.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _origins,
    allow_credentials = False,   # must be False when using wildcard or simple origins
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],
    expose_headers    = ["*"],
)

# Constants

# 100 MB hard limit — Render free tier has limited memory; reject early
MAX_FILE_BYTES = 100 * 1024 * 1024

# Only these MIME types are accepted — blocks images, PDFs, etc.
ALLOWED_MIME_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}

# Temp upload directory — files land here, get processed, then get deleted
UPLOAD_DIR = Path("/tmp/socialmedia_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# How long (seconds) to wait before deleting a processed file
FILE_TTL_SECONDS = 300   # 5 minutes

# Lazy model imports
# We import inference modules only when needed so the server starts fast.
# Heavy models (EfficientNet, XGBoost, LightGBM) are loaded on first call
# and then cached in memory for subsequent requests (see _load_models() in each file).

def _get_model_a():
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from model_a import inference as model_a_inference
    return model_a_inference

def _get_model_b():
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from model_b import inference as model_b_inference
    return model_b_inference

def _get_reports():
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from llm.reports import forensic_report, virality_report
    return forensic_report, virality_report

# Helpers

async def _save_upload(file: UploadFile) -> Path:
    """
    Validates the uploaded file (size + MIME type) and saves it to UPLOAD_DIR.
    Returns the saved file path.
    Raises HTTPException on any validation failure.
    """
    # Check MIME type first — cheap check before reading bytes
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail      = f"Unsupported file type '{file.content_type}'. Only video files are accepted.",
        )

    # Read the file in chunks and enforce the 100 MB size cap
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    save_path   = UPLOAD_DIR / unique_name

    total_bytes = 0
    with open(save_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):   # 1 MB chunks
            total_bytes += len(chunk)
            if total_bytes > MAX_FILE_BYTES:
                out.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail      = "File exceeds the 100 MB limit.",
                )
            out.write(chunk)

    return save_path


async def _schedule_delete(file_path: Path, delay: int = FILE_TTL_SECONDS):
    """
    Waits `delay` seconds then deletes the file.
    Runs as a background asyncio task — non-blocking.
    """
    await asyncio.sleep(delay)
    file_path.unlink(missing_ok=True)


# Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — prevents 404 on Render's internal health checks."""
    return {"status": "ok", "service": "SynthSenses API"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness check — used by Render.com to verify the server is up."""
    return {"status": "ok"}



@app.post("/analyze/synthetic", tags=["Analysis"])
@limiter.limit("10/minute")
async def analyze_synthetic(
    request: Request,                        # required by slowapi for rate limiting
    video:   UploadFile = File(...),
):
    """
    Detects whether the uploaded video is Real, a Deepfake, or AI-Generated.
    Returns the label, confidence score, and per-class probabilities.
    The uploaded file is deleted automatically after 5 minutes.
    """
    video_path = await _save_upload(video)
    asyncio.create_task(_schedule_delete(video_path))

    try:
        model_a = _get_model_a()
        result  = model_a.predict(video_path)
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Model A inference failed: {str(e)}")

    try:
        forensic_report, _ = _get_reports()
        explanation = forensic_report(result)
    except Exception:
        explanation = ""

    return JSONResponse(content={
        "label":          result["label"],
        "confidence":     result["confidence"],
        "prob_ai":        result["prob_ai"],
        "prob_deepfake":  result["prob_deepfake"],
        "explanation":    explanation,
    })


@app.post("/analyze/virality", tags=["Analysis"])
@limiter.limit("10/minute")
async def analyze_virality(
    request:       Request,
    video:         UploadFile = File(...),
    title:         str        = Form(""),
    post_hour:     int        = Form(12),     # default: noon
    post_day:      int        = Form(1),      # default: Tuesday (0=Monday)
    tag_count:     int        = Form(5),
    user_caption:  str        = Form(""),
    user_hashtags: str        = Form(""),
):
    """
    Predicts whether the uploaded video is likely to go viral.
    Returns a virality score (0-100), label (Viral / Not Viral),
    probability, top 5 influential features, and all extracted feature values.
    The uploaded file is deleted automatically after 5 minutes.
    """
    # Validate form inputs — prevents garbage from reaching the model
    if not (0 <= post_hour <= 23):
        raise HTTPException(status_code=400, detail="post_hour must be between 0 and 23.")
    if not (0 <= post_day <= 6):
        raise HTTPException(status_code=400, detail="post_day must be between 0 (Monday) and 6 (Sunday).")
    if tag_count < 0:
        raise HTTPException(status_code=400, detail="tag_count cannot be negative.")

    video_path = await _save_upload(video)
    asyncio.create_task(_schedule_delete(video_path))

    try:
        model_b = _get_model_b()
        result  = model_b.predict(
            video_path = video_path,
            title      = title,
            post_hour  = post_hour,
            post_day   = post_day,
            tag_count  = tag_count,
        )
    except Exception as e:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Model B inference failed: {str(e)}")

    try:
        _, virality_report = _get_reports()
        explanation = virality_report(result, user_caption, user_hashtags)
    except Exception:
        explanation = ""

    return JSONResponse(content={
        "virality_score": result["virality_score"],
        "label":          result["label"],
        "probability":    result["probability"],
        "top_features":   result["top_features"],
        "features":       result["features"],
        "explanation":    explanation,
    })
