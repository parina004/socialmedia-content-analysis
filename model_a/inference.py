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
    _sub1 = xgb.XGBClassifier()  # real vs deepfake (2064 features)
    _sub1.load_model(str(ROOT / "submodel_df.json"))

    _sub2 = xgb.XGBClassifier()  # real vs AI-generated (2048 features)
    _sub2.load_model(str(ROOT / "submodel_ai.json"))


def _extract_frames(video_path: Path, tmp_dir: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_seconds = min(int(total_frames / fps), 30)  # cap at 30 frames

    for s in range(n_seconds):
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

    # 1. Extract frames to a temp folder
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        _extract_frames(video_path, tmp_dir)

        # 2. Run all 3 streams → 2064-dim vector
        feature_vec = extract_video_features(tmp_dir, _efficientnet, _forensic_cnn)
    finally:
        shutil.rmtree(tmp_dir)  # always clean up, even if something crashes

    # 3. Reshape to (1, 2064) — model expects a batch
    X = feature_vec.reshape(1, -1)

    # 4. Get raw probabilities from both submodels
    prob_ai = float(_sub2.predict_proba(X[:, :2048])[:, 1][0])
    prob_df = float(_sub1.predict_proba(X)[:, 1][0])
    # 5. Apply cascade thresholds
    if prob_ai >= AI_THRESH:
        label_idx = 2  # AI-Generated
        confidence = prob_ai
    elif prob_df >= DF_THRESH:
        label_idx = 1  # Deepfake
        confidence = prob_df
    else:
        label_idx = 0  # Real
        confidence = 1.0 - max(prob_ai, prob_df)

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