import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from data.model_b_datasets.extract_features import (
    extract_visual_features,
    extract_audio_features,
    extract_metadata_features,
)

FEATURE_COLS = [
    "brisque_score", "color_vibrancy", "motion_intensity",
    "face_presence_ratio", "face_emotion_joy", "face_emotion_surprise",
    "thumbnail_brightness",
    "tempo_bpm", "rms_energy", "speech_ratio", "zero_crossing_rate", "beat_strength",
    "title_sentiment", "title_length", "title_has_question", "title_has_number",
    "description_length", "tag_count", "upload_hour", "upload_day",
    "like_to_view_ratio", "comment_to_view_ratio",
]

MEAN_LIKE_RATIO = 0.04      # ~4% of viewers like — typical YouTube average
MEAN_COMMENT_RATIO = 0.005  # ~0.5% of viewers comment — typical YouTube average

_model = None

def _load_model():
    global _model
    if _model is None:
        with open(Path(__file__).parent / "model.pkl", "rb") as f:
            _model = pickle.load(f)


def predict(video_path: Path, title: str, post_hour: int, post_day: int, tag_count: int) -> dict:
    _load_model()

    visual = extract_visual_features(video_path)
    audio = extract_audio_features(video_path)

    row = {
        "title": title,
        "description": "",
        "tags": "|".join(["tag"] * tag_count),
        "upload_date": f"2024-01-01T{post_hour:02d}:00:00",
    }
    metadata = extract_metadata_features(row)

    all_features = {
        **visual,
        **audio,
        **metadata,
        "like_to_view_ratio": MEAN_LIKE_RATIO,
        "comment_to_view_ratio": MEAN_COMMENT_RATIO,
    }

    df = pd.DataFrame([all_features])[FEATURE_COLS]

    prob = float(_model.predict_proba(df)[0][1])
    virality_score = round(prob * 100)

    importances = dict(zip(FEATURE_COLS, _model.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "virality_score": virality_score,
        "label": "Viral" if prob >= 0.5 else "Not Viral",
        "probability": round(prob, 4),
        "top_features": {k: round(float(v), 4) for k, v in top5},
    }

#standalone test
if __name__ == "__main__":
    result = predict(
        video_path=Path(r"C:\Users\parin\Downloads\testingvideo.mp4"),
        title="Test Video Title",
        post_hour=15,
        post_day=2,
        tag_count=5,
    )
    print(result)

##virality score wont be perfect due to this random test vid with generic title, no description and only 5 fake tags
## try with inputs optimized for virality later

