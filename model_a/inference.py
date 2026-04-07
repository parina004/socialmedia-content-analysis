import logging
import tempfile
import shutil
from pathlib import Path
import numpy as np
import xgboost as xgb
import cv2
import sys
import torch

sys.path.append(str(Path(__file__).parent.parent))
from model_a.extract_features import extract_video_features, build_efficientnet, ForensicCNN, DEVICE

log = logging.getLogger("model_a")

#thresholds from train file
AI_THRESH = 0.50
DF_THRESH = 0.50
LABEL_MAP = {0: "Real", 1: "Deepfake", 2: "AI-Generated"}

_efficientnet = None
_forensic_cnn = None
_sub1 = None
_sub2 = None

def _load_models():
    global _efficientnet, _forensic_cnn, _sub1, _sub2
    if _sub1 is not None:
        return  # already loaded, skip

    ROOT = Path(__file__).parent

    # Load EfficientNet-B4
    _efficientnet = build_efficientnet()

    # Load ForensicCNN
    _forensic_cnn = ForensicCNN().to(DEVICE)
    _forensic_cnn.load_state_dict(
        torch.load(ROOT / "forensic_cnn.pth", map_location=DEVICE)
    )
    _forensic_cnn.eval()

    # Load XGBoost models
    _sub1 = xgb.XGBClassifier()
    _sub1.load_model(str(ROOT / "submodel1.json"))

    _sub2 = xgb.XGBClassifier()
    _sub2.load_model(str(ROOT / "submodel2.json"))


MAX_FRAMES = 8  # sample 8 evenly-spaced frames — enough for reliable detection, much faster on CPU

def _extract_frames(video_path: Path, tmp_dir: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    # Pick MAX_FRAMES evenly-spaced second-marks across the video (max 30s)
    sample_secs = [
        int(i * min(duration, 30) / (MAX_FRAMES - 1))
        for i in range(MAX_FRAMES)
    ]

    for s in sample_secs:
        frame_idx = int(s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        resized = cv2.resize(frame, (224, 224))
        cv2.imwrite(str(tmp_dir / f"frame_{s:04d}.jpg"), resized)

    cap.release()
    return tmp_dir

def predict(video_path: Path) -> dict:
    _load_models()

    log.info(f"Extracting frames from: {video_path.name}")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        _extract_frames(video_path, tmp_dir)
        n_frames = len(list(tmp_dir.glob("*.jpg")))
        log.info(f"Extracted {n_frames} frames — running 3-stream feature extraction...")
        feature_vec = extract_video_features(tmp_dir, _efficientnet, _forensic_cnn)
    finally:
        shutil.rmtree(tmp_dir)

    X = feature_vec.reshape(1, -1)

    log.info("Submodel 2: checking if AI-Generated...")
    prob_ai = float(_sub2.predict_proba(X[:, :2048])[:, 1][0])
    log.info(f"  AI-Generated probability: {prob_ai:.4f}")

    log.info("Submodel 1: checking if Deepfake...")
    prob_df = float(_sub1.predict_proba(X)[:, 1][0])
    log.info(f"  Deepfake probability: {prob_df:.4f}")

    if prob_ai >= AI_THRESH:
        label_idx = 2
        confidence = prob_ai
    elif prob_df >= DF_THRESH:
        label_idx = 1
        confidence = prob_df
    else:
        label_idx = 0
        confidence = 1.0 - max(prob_ai, prob_df)

    log.info(f"Final verdict: {LABEL_MAP[label_idx]} (confidence {confidence:.2%})")

    return {
        "label": LABEL_MAP[label_idx],
        "confidence": round(confidence, 4),
        "prob_ai": round(prob_ai, 4),
        "prob_deepfake": round(prob_df, 4),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(r"Usage: python model_a/inference.py C:\Users\parin\Downloads\testingvideo.mp4")
        sys.exit(1)
    result = predict(Path(sys.argv[1]))
    print(result)