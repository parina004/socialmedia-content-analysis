"""
model_a/predict.py
Loads the trained Model A cascade and runs inference on a single video file.
Returns: label (Real / Deepfake / AI-Generated), confidence %, and per-stream scores.

Pipeline:
  Video → extract 1 frame/sec → run 3 streams → average → cascade XGBoost → label

Run with: uv run model_a/predict.py <path_to_video>
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import scipy.fft
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "model_a"
CNN_PATH   = MODEL_DIR / "forensic_cnn.pth"
SM1_PATH   = MODEL_DIR / "submodel1.json"
SM2_PATH   = MODEL_DIR / "submodel2.json"

## Cascade thresholds tuned during training
AI_THRESH = 0.65
DF_THRESH = 0.40

## Label names
LABELS = {0: "Real", 1: "Deepfake", 2: "AI-Generated"}

## Device — prefer Apple MPS, then CUDA, then CPU
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)

IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
TO_TENSOR = transforms.ToTensor()

## SRM high-pass noise residual filters — expose noise patterns left by deepfake generators
_SRM_K = [
    torch.tensor([[0, 0, 0], [0.5, -1, 0.5], [0, 0, 0]],            dtype=torch.float32),
    torch.tensor([[0, 0.5, 0], [0, -1, 0], [0, 0.5, 0]],             dtype=torch.float32),
    torch.tensor([[0.25, -0.5, 0.25], [-0.5, 1, -0.5], [0.25, -0.5, 0.25]], dtype=torch.float32),
]
SRM_KERNELS = torch.stack([k.unsqueeze(0).unsqueeze(0) for k in _SRM_K]).to(DEVICE)


## The Forensic CNN definition — must match exactly what was used during training
class ForensicCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        ## Returns both logits and the 256-dim feature vector (we only use the features)
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.head(feat), feat


def _srm_residuals(gray: torch.Tensor) -> torch.Tensor:
    ## Applies 3 SRM filters to a grayscale tensor → (B, 3, H, W)
    if gray.dim() == 3:
        gray = gray.unsqueeze(0)
    return torch.cat([F.conv2d(gray, k, padding=1) for k in SRM_KERNELS.unbind(0)], dim=1)


def _dct_channels(img_np: np.ndarray) -> np.ndarray:
    ## Converts a (H, W, 3) BGR image to 3 DCT frequency channels
    out = []
    for c in range(3):
        ch  = img_np[:, :, c].astype(np.float32) / 255.0
        dct = np.abs(scipy.fft.dctn(ch, norm="ortho"))
        dct = np.log1p(dct)
        lo, hi = dct.min(), dct.max()
        dct = (dct - lo) / (hi - lo + 1e-8)
        out.append(dct)
    return np.stack(out).astype(np.float32)


def _forensic_tensor(img_np: np.ndarray) -> torch.Tensor:
    ## Builds the 6-channel forensic input: 3 DCT channels + 3 SRM noise channels
    rgb  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    dct  = torch.tensor(_dct_channels(rgb))
    gray = torch.tensor(
        cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)
    srm  = _srm_residuals(gray.to(DEVICE)).squeeze(0).cpu()
    return torch.cat([dct, srm], dim=0)  ## (6, H, W)


## MediaPipe landmark indices — same 16 key points used during training
_LM = {
    "left_eye_outer": 33,  "right_eye_outer": 263,
    "left_eye_inner": 133, "right_eye_inner": 362,
    "nose_tip":  1,        "nose_bridge":  168,
    "left_mouth":  61,     "right_mouth":  291,
    "upper_lip":  13,      "lower_lip":  14,
    "chin":  18,           "forehead":  10,
    "left_cheek":  234,    "right_cheek":  454,
    "left_brow_outer":  70,"right_brow_outer": 300,
}

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
)


def _geo_features(img_np: np.ndarray) -> np.ndarray:
    ## Extracts 16 facial geometry ratios from a frame; returns zero vector if no face
    H, W = img_np.shape[:2]
    rgb  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    res  = _face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return np.zeros(16, dtype=np.float32)

    lm = res.multi_face_landmarks[0].landmark
    L  = _LM

    def d(a, b):
        pa = np.array([lm[a].x * W, lm[a].y * H])
        pb = np.array([lm[b].x * W, lm[b].y * H])
        return float(np.linalg.norm(pa - pb))

    face_h  = d(L["forehead"],        L["chin"])           + 1e-6
    face_w  = d(L["left_cheek"],      L["right_cheek"])    + 1e-6
    eye_d   = d(L["left_eye_outer"],  L["right_eye_outer"])
    mouth_w = d(L["left_mouth"],      L["right_mouth"])
    mouth_h = d(L["upper_lip"],       L["lower_lip"])
    l_eye_w = d(L["left_eye_outer"],  L["left_eye_inner"])
    r_eye_w = d(L["right_eye_outer"], L["right_eye_inner"])
    brow_l  = abs(lm[L["left_brow_outer"]].y  - lm[L["left_eye_outer"]].y)  * H
    brow_r  = abs(lm[L["right_brow_outer"]].y - lm[L["right_eye_outer"]].y) * H
    chin_m  = d(L["chin"],            L["lower_lip"])
    nose_h  = d(L["nose_bridge"],     L["nose_tip"])
    nose_w  = mouth_w * 0.6
    mid_x   = (lm[L["left_cheek"]].x + lm[L["right_cheek"]].x) / 2 * W
    sym     = abs(lm[L["nose_tip"]].x * W - mid_x) / (face_w + 1e-6)

    return np.clip(np.array([
        eye_d   / face_h,
        l_eye_w / face_h,
        r_eye_w / face_h,
        nose_w  / face_w,
        nose_h  / face_h,
        mouth_w / face_w,
        mouth_h / face_h,
        face_w  / face_h,
        brow_l  / face_h,
        brow_r  / face_h,
        chin_m  / face_h,
        abs(l_eye_w - r_eye_w) / (l_eye_w + r_eye_w + 1e-6),
        abs(brow_l  - brow_r)  / (face_h + 1e-6),
        sym,
        nose_w  / face_h,
        eye_d   / face_w,
    ], dtype=np.float32), 0.0, 5.0)


## Singleton model cache — load once, reuse for every call
_models: dict = {}


def _load_models():
    ## Load all models from disk once and cache them in memory
    if _models:
        return _models

    ## EfficientNet-B4 — pretrained backbone, classification head removed
    eff = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0, global_pool="avg")
    ## We use pretrained weights here because EfficientNet was frozen during training
    eff = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0, global_pool="avg")
    for p in eff.parameters():
        p.requires_grad = False
    eff.eval()
    _models["efficientnet"] = eff.to(DEVICE)

    ## Forensic CNN — load trained weights from disk
    cnn = ForensicCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE, weights_only=True))
    cnn.eval()
    _models["forensic_cnn"] = cnn

    ## XGBoost cascade — submodel 2 (AI-Generated?) runs first, submodel 1 (Deepfake?) runs second
    sub2 = xgb.XGBClassifier()
    sub2.load_model(str(SM2_PATH))
    _models["sub2"] = sub2

    sub1 = xgb.XGBClassifier()
    sub1.load_model(str(SM1_PATH))
    _models["sub1"] = sub1

    return _models


def _extract_frames(video_path: str) -> list[np.ndarray]:
    ## Sample 1 frame per second from the video — same rate used during training
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step   = max(1, int(fps))  ## skip this many frames between samples
    frames = []
    idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


@torch.no_grad()
def predict(video_path: str) -> dict:
    """
    Run Model A inference on a video file.

    Args:
        video_path: Path to the video file.

    Returns a dict with:
        label        : "Real" | "Deepfake" | "AI-Generated"
        confidence   : float, 0–100 (probability of the predicted class × 100)
        prob_ai      : float, Submodel 2's probability that the video is AI-generated
        prob_deepfake: float, Submodel 1's probability that the video is a deepfake
        face_score   : float or None — mean geometric score; None if no face detected
    """
    m = _load_models()

    ## Strip any surrounding quotes the agent sometimes adds to path strings
    video_path = video_path.strip().strip("'\"")
    frames = _extract_frames(video_path)
    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")

    s1_vecs, s2_vecs, s3_vecs = [], [], []

    for frame in frames:
        ## Stream 1 — EfficientNet-B4 RGB features (1792-dim)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        t     = IMAGENET_NORM(TO_TENSOR(pil)).unsqueeze(0).to(DEVICE)
        s1    = m["efficientnet"](t).squeeze().cpu().numpy()
        s1_vecs.append(s1)

        ## Stream 2 — Forensic CNN (DCT + SRM) features (256-dim)
        ft    = _forensic_tensor(frame).unsqueeze(0).to(DEVICE)
        _, s2 = m["forensic_cnn"](ft)
        s2_vecs.append(s2.squeeze().cpu().numpy())

        ## Stream 3 — MediaPipe facial geometry (16-dim)
        s3_vecs.append(_geo_features(frame))

    ## Average all frame-level vectors into a single per-video vector
    s1_mean = np.mean(s1_vecs, axis=0)  ## (1792,)
    s2_mean = np.mean(s2_vecs, axis=0)  ## (256,)
    s3_mean = np.mean(s3_vecs, axis=0)  ## (16,)

    ## Concatenate: 1792 + 256 + 16 = 2064-dim
    feature_vec = np.concatenate([s1_mean, s2_mean, s3_mean]).astype(np.float32).reshape(1, -1)

    ## Cascade: Submodel 2 first (AI-Generated?), then Submodel 1 (Deepfake?) if not AI
    prob_ai = float(m["sub2"].predict_proba(feature_vec[:, :2048])[:, 1][0])
    prob_df = float(m["sub1"].predict_proba(feature_vec)[:, 1][0])

    if prob_ai >= AI_THRESH:
        label      = "AI-Generated"
        confidence = round(prob_ai * 100, 2)
    elif prob_df >= DF_THRESH:
        label      = "Deepfake"
        confidence = round(prob_df * 100, 2)
    else:
        label      = "Real"
        confidence = round((1.0 - max(prob_ai, prob_df)) * 100, 2)

    ## Face score: mean of geometric features; None if all zeros (no face detected)
    face_score = float(np.mean(s3_mean)) if np.any(s3_mean != 0) else None

    return {
        "label":         label,
        "confidence":    confidence,
        "prob_ai":       round(prob_ai * 100, 4),
        "prob_deepfake": round(prob_df * 100, 4),
        "face_score":    round(face_score, 4) if face_score is not None else None,
    }


## Quick test: uv run model_a/predict.py path/to/video.mp4
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run model_a/predict.py <video_path>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = predict(sys.argv[1])
    print("\nModel A — Synthetic Detection Result")
    print("=" * 40)
    for k, v in result.items():
        print(f"  {k:<20}: {v}")
