# llm/reports.py
# Standalone report generators called directly by the API endpoints.
# Each function takes the model's output dict, fetches RAG context,
# formats the prompt, and returns the LLM's plain-text explanation.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rag.retriever import retrieve
from llm.llm_client import generate
from llm.prompts import format_synthetic_prompt, format_virality_prompt


def forensic_report(result: dict) -> str:
    """
    Generates a plain-text forensic explanation for a synthetic detection result.
    result must contain: label, confidence, prob_ai, prob_deepfake
    """
    label      = result["label"]
    confidence = float(result["confidence"])
    prob_ai    = float(result["prob_ai"])
    prob_df    = float(result["prob_deepfake"])

    # Choose a RAG query that matches what was actually detected
    if label == "AI-Generated":
        query = "AI video generation artifacts diffusion model detection"
    elif label == "Deepfake":
        query = "deepfake facial manipulation detection forensic evidence"
    elif confidence < 0.65:
        query = "borderline inconclusive synthetic media detection threshold"
    else:
        query = "real authentic video detection synthetic media"

    rag_context = retrieve(query, k=5)

    prompt = format_synthetic_prompt(
        label              = label,
        confidence         = confidence * 100,  # prompt expects percentage e.g. 91.5
        efficientnet_score = prob_ai,
        forensic_score     = prob_df,
        face_score         = None,              # not exposed at API level
        rag_context        = rag_context,
    )

    return generate(prompt)


def virality_report(result: dict, user_caption: str = "", user_hashtags: str = "") -> str:
    """
    Generates a plain-text virality analysis with actionable suggestions.
    result must contain: label, probability, virality_score, features
    """
    f = result["features"]

    day_names  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    upload_day = day_names[int(f.get("upload_day", 0))]

    rag_context = retrieve("virality prediction social media engagement features", k=5)

    prompt = format_virality_prompt(
        virality_label        = result["label"],
        viral_probability     = result["probability"] * 100,
        engagement_percentile = float(result["virality_score"]),
        brisque               = float(f.get("brisque_score", 0.0)),
        vibrancy              = float(f.get("color_vibrancy", 0.0)),
        motion                = float(f.get("motion_intensity", 0.0)),
        face_ratio            = float(f.get("face_presence_ratio", 0.0)),
        tempo                 = float(f.get("tempo_bpm", 0.0)),
        rms_energy            = float(f.get("rms_energy", 0.0)),
        speech_ratio          = float(f.get("speech_ratio", 0.0)),
        title_sentiment       = float(f.get("title_sentiment", 0.0)),
        title_length          = int(f.get("title_length", 0)),
        tag_count             = int(f.get("tag_count", 0)),
        upload_hour           = int(f.get("upload_hour", 0)),
        upload_day            = upload_day,
        user_caption          = user_caption,
        user_hashtags         = user_hashtags,
        rag_context           = rag_context,
    )

    return generate(prompt)
