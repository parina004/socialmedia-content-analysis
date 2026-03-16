"""
model_b/predict.py
Loads the trained Model B LightGBM classifier and runs inference on a single video.
Analyses only the first 60 seconds of the video.

Returns: virality_label, viral_probability, engagement_percentile, and all 20 features.

Run with: uv run model_b/predict.py <path_to_video> [--title "..."] [--hour 18] [--day Friday] [--tags 5]
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
import logging
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
from textblob import TextBlob

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "model_b" / "model.pkl"

## Only analyse the first 60 seconds — same constraint used during training
MAX_SECONDS = 60

## Dataset means for engagement ratios — used at inference since we have no real data yet
MEAN_LIKE_RATIO    = 0.028
MEAN_COMMENT_RATIO = 0.0021

## Day-of-week encoding — must match what was used during training (0=Monday … 6=Sunday)
DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

## Feature column order — must exactly match what the model was trained on
FEATURE_COLS = [
    "brisque_score", "color_vibrancy", "motion_intensity",
    "face_presence_ratio", "face_emotion_joy", "face_emotion_surprise",
    "thumbnail_brightness",
    "tempo_bpm", "rms_energy", "speech_ratio", "zero_crossing_rate", "beat_strength",
    "title_sentiment", "title_length", "title_has_question", "title_has_number",
    "description_length", "tag_count", "upload_hour", "upload_day",
    "like_to_view_ratio", "comment_to_view_ratio",
]

## Singleton cache
_model = None


def _load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _extract_visual_features(video_path: str) -> dict:
    """
    Extract 7 visual features from the first 60 seconds of the video.
    Uses OpenCV for frame analysis and DeepFace for emotion detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = int(fps * MAX_SECONDS)

    frames    = []
    brightnesses = []
    vibranices   = []
    prev_gray    = None
    motion_vals  = []

    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        ## Sample 1 frame per second for efficiency
        if idx % max(1, int(fps)) == 0:
            frames.append(frame)

            ## Thumbnail brightness — mean value in HSV space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightnesses.append(float(hsv[:, :, 2].mean()) / 255.0)

            ## Color vibrancy — mean saturation in HSV space
            vibranices.append(float(hsv[:, :, 1].mean()) / 255.0)

            ## Motion intensity — mean absolute diff between consecutive gray frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if prev_gray is not None:
                motion_vals.append(float(np.mean(np.abs(gray - prev_gray))))
            prev_gray = gray

        idx += 1

    cap.release()

    if not frames:
        ## Return zeros if no frames could be read
        return {k: 0.0 for k in [
            "brisque_score", "color_vibrancy", "motion_intensity",
            "face_presence_ratio", "face_emotion_joy", "face_emotion_surprise",
            "thumbnail_brightness",
        ]}

    ## BRISQUE quality score — lower is better quality (range ~0–100)
    ## Use piq library; fall back to 50 (neutral) if unavailable
    try:
        import piq
        import torch
        from torchvision import transforms
        rgb_tensor = transforms.ToTensor()(
            cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        ).unsqueeze(0).clamp(0, 1)
        brisque_val = float(piq.brisque(rgb_tensor, data_range=1.0))
    except Exception:
        brisque_val = 50.0

    ## Face presence and emotions — use DeepFace on sampled frames
    face_count    = 0
    joy_scores    = []
    surprise_scores = []

    try:
        from deepface import DeepFace
        for frame in frames[::3]:  ## Check every 3rd frame to save time
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=False, silent=True,
                )
                emotions = result[0]["emotion"] if isinstance(result, list) else result["emotion"]
                ## DeepFace returns emotion probabilities as 0–100
                joy_scores.append(emotions.get("happy", 0.0) / 100.0)
                surprise_scores.append(emotions.get("surprise", 0.0) / 100.0)
                if result[0]["dominant_emotion"] if isinstance(result, list) else result["dominant_emotion"]:
                    face_count += 1
            except Exception:
                pass
    except ImportError:
        pass

    total_checked = max(len(frames[::3]), 1)
    face_presence = face_count / total_checked
    joy_mean      = float(np.mean(joy_scores))      if joy_scores else 0.0
    surprise_mean = float(np.mean(surprise_scores)) if surprise_scores else 0.0

    return {
        "brisque_score":       round(brisque_val, 4),
        "color_vibrancy":      round(float(np.mean(vibranices)), 4),
        "motion_intensity":    round(float(np.mean(motion_vals)) if motion_vals else 0.0, 4),
        "face_presence_ratio": round(face_presence, 4),
        "face_emotion_joy":    round(joy_mean, 4),
        "face_emotion_surprise": round(surprise_mean, 4),
        "thumbnail_brightness": round(float(np.mean(brightnesses)), 4),
    }


def _extract_audio_features(video_path: str) -> dict:
    """
    Extract 5 audio features from the first 60 seconds of the video.
    Uses ffmpeg to pull audio to a temp WAV file, then librosa for analysis.
    """
    ## Default values in case audio extraction fails
    defaults = {
        "tempo_bpm": 0.0, "rms_energy": 0.0, "speech_ratio": 0.0,
        "zero_crossing_rate": 0.0, "beat_strength": 0.0,
    }

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        ## Extract audio for first MAX_SECONDS seconds using ffmpeg
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-t", str(MAX_SECONDS),
                "-ac", "1",       ## mono
                "-ar", "22050",   ## sample rate librosa expects
                "-vn",            ## no video
                tmp_path,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )

        ## Load audio with librosa
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        Path(tmp_path).unlink(missing_ok=True)

        if len(y) == 0:
            return defaults

        ## Tempo and beat strength
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        onset_env  = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = float(np.mean(onset_env)) if len(onset_env) > 0 else 0.0

        ## Energy
        rms = librosa.feature.rms(y=y)
        rms_energy = float(np.mean(rms))

        ## Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = float(np.mean(zcr))

        ## Speech ratio — estimate from spectral centroid (speech typically 300–3000 Hz)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        ## Frames where centroid is in speech range (~300–3500 Hz)
        speech_frames = np.sum((centroid >= 300) & (centroid <= 3500))
        speech_ratio  = float(speech_frames) / max(centroid.shape[1], 1)

        return {
            "tempo_bpm":          round(float(tempo), 2),
            "rms_energy":         round(rms_energy, 6),
            "speech_ratio":       round(speech_ratio, 4),
            "zero_crossing_rate": round(zcr_mean, 6),
            "beat_strength":      round(beat_strength, 4),
        }

    except Exception as e:
        log.warning("Audio extraction failed: %s — using zeros", e)
        return defaults


def _metadata_features(
    title: str,
    description: str,
    tag_count: int,
    upload_hour: int,
    upload_day: str,
) -> dict:
    """
    Compute 8 metadata features from user-provided content details.
    """
    sentiment = TextBlob(title).sentiment.polarity if title else 0.0
    day_num   = DAY_MAP.get(upload_day.lower(), 3)  ## default Wednesday if unknown

    return {
        "title_sentiment":    round(sentiment, 4),
        "title_length":       len(title),
        "title_has_question": int("?" in title),
        "title_has_number":   int(any(c.isdigit() for c in title)),
        "description_length": len(description),
        "tag_count":          tag_count,
        "upload_hour":        upload_hour,
        "upload_day":         day_num,
    }


def predict(
    video_path: str,
    title: str = "",
    description: str = "",
    tag_count: int = 5,
    upload_hour: int = 12,
    upload_day: str = "Monday",
) -> dict:
    """
    Run Model B inference on a video file.

    Args:
        video_path  : Path to the video file.
        title       : Planned video title (used for metadata features).
        description : Planned description (used for description_length feature).
        tag_count   : Number of tags the user plans to add.
        upload_hour : Hour of day the video will be posted (0–23).
        upload_day  : Day of week the video will be posted (e.g. "Friday").

    Returns a dict with:
        virality_label       : "Viral" | "Not Viral"
        viral_probability    : float, 0–100
        engagement_percentile: float, estimated percentile
        features             : dict of all 20 extracted features
    """
    ## Strip any surrounding quotes the agent sometimes adds to path strings
    video_path = video_path.strip().strip("'\"")

    model = _load_model()

    ## Extract all feature groups
    visual   = _extract_visual_features(video_path)
    audio    = _extract_audio_features(video_path)
    metadata = _metadata_features(title, description, tag_count, upload_hour, upload_day)

    ## Engagement ratios — set to dataset mean since we have no real data at inference time
    engagement = {
        "like_to_view_ratio":    MEAN_LIKE_RATIO,
        "comment_to_view_ratio": MEAN_COMMENT_RATIO,
    }

    ## Combine all features into one dict
    all_features = {**visual, **audio, **metadata, **engagement}

    ## Build the feature row in the exact column order the model was trained on
    row = pd.DataFrame([[all_features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)

    ## Run inference
    viral_prob  = float(model.predict_proba(row)[:, 1][0])
    label       = "Viral" if viral_prob >= 0.5 else "Not Viral"

    ## Approximate engagement percentile from viral probability
    percentile  = round(viral_prob * 100, 1)

    return {
        "virality_label":        label,
        "viral_probability":     round(viral_prob * 100, 2),
        "engagement_percentile": percentile,
        "features":              all_features,
    }


## Quick test: uv run model_b/predict.py path/to/video.mp4 --title "My video" --hour 18 --day Friday
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--title",       default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--tags",        type=int, default=5)
    parser.add_argument("--hour",        type=int, default=12)
    parser.add_argument("--day",         default="Monday")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = predict(
        args.video_path,
        title=args.title,
        description=args.description,
        tag_count=args.tags,
        upload_hour=args.hour,
        upload_day=args.day,
    )

    print("\nModel B — Virality Prediction Result")
    print("=" * 40)
    print(f"  Label               : {result['virality_label']}")
    print(f"  Viral Probability   : {result['viral_probability']}%")
    print(f"  Engagement Percentile: {result['engagement_percentile']}th")
    print("\n  Extracted Features:")
    for k, v in result["features"].items():
        print(f"    {k:<30}: {v}")
