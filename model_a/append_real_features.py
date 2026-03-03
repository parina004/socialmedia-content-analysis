## Model A — Append Real Features  (run after collect_youtube_real.py)

## Extracts features ONLY for the newly collected youtube_interviews videos
## and appends them to the existing features_train.npz / features_test.npz.
## The existing ~17,000 videos are NOT reprocessed — this is intentional to
## avoid re-running a multi-hour job when we only added one new source.

## Uses the already-trained forensic_cnn.pth — does NOT retrain the CNN.
## The ForensicCNN was trained on the original data; we reuse it as-is for
## the new real clips (adding real clips doesn't change the CNN's behaviour
## since Stream 2 is frozen when we run inference here).

## After this script finishes:
##   1. Delete model_a/.sm1_checkpoint.json  (forces Submodel 1 to retrain)
##   2. Run model_a/train.py                 (Submodel 2 is skipped, Submodel 1 retrains
## Run with:  uv run model_a/append_real_features.py

import csv
import json
import logging
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import scipy.fft
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
MANIFEST = ROOT / "data" / "model_a_datasets" / "frames" / "manifest.csv"
OUT_DIR  = ROOT / "model_a"
CNN_PATH  = OUT_DIR / "forensic_cnn.pth"
TRAIN_NPZ = OUT_DIR / "features_train.npz"
TEST_NPZ  = OUT_DIR / "features_test.npz"
CKPT_PATH = OUT_DIR / ".append_checkpoint.json"

## only process rows from this dataset source
TARGET_SOURCE = "youtube_interviews"

## pause briefly every N videos to rest the system
PAUSE_EVERY = 200
PAUSE_SECS  = 10

LABEL_MAP = {"real": 0, "deepfake": 1, "ai_generated": 2}

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)

IMAGENET_NORM = __import__("torchvision").transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
TO_TENSOR = __import__("torchvision").transforms.ToTensor()


## SRM and DCT helpers — same logic as extract_features.py, copied here so
## this script can run standalone without importing from that module.

_SRM_K = [
    torch.tensor([[0, 0, 0], [0.5, -1, 0.5], [0, 0, 0]],           dtype=torch.float32),
    torch.tensor([[0, 0.5, 0], [0, -1, 0], [0, 0.5, 0]],           dtype=torch.float32),
    torch.tensor([[0.25, -0.5, 0.25], [-0.5, 1, -0.5], [0.25, -0.5, 0.25]], dtype=torch.float32),
]
SRM_KERNELS = torch.stack([k.unsqueeze(0).unsqueeze(0) for k in _SRM_K]).to(DEVICE)


def srm_residuals(gray_tensor: torch.Tensor) -> torch.Tensor:
    if gray_tensor.dim() == 3:
        gray_tensor = gray_tensor.unsqueeze(0)
    return torch.cat([F.conv2d(gray_tensor, k, padding=1) for k in SRM_KERNELS.unbind(0)], dim=1)


def compute_dct_channels(img_np: np.ndarray) -> np.ndarray:
    out = []
    for c in range(3):
        ch  = img_np[:, :, c].astype(np.float32) / 255.0
        dct = np.abs(scipy.fft.dctn(ch, norm="ortho"))
        dct = np.log1p(dct)
        lo, hi = dct.min(), dct.max()
        out.append(((dct - lo) / (hi - lo + 1e-8)).astype(np.float32))
    return np.stack(out)


def frame_to_forensic_tensor(img_np: np.ndarray) -> torch.Tensor:
    rgb  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    dct  = torch.tensor(compute_dct_channels(rgb))
    gray = torch.tensor(
        cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)
    srm = srm_residuals(gray.to(DEVICE)).squeeze(0).cpu()
    return torch.cat([dct, srm], dim=0)


## MediaPipe landmark indices and geometric feature computation.
## Produces a 16-dim vector of facial ratios per frame.

_LM = {
    "left_eye_outer": 33,  "right_eye_outer": 263,
    "left_eye_inner": 133, "right_eye_inner": 362,
    "nose_tip": 1,         "nose_bridge": 168,
    "left_mouth": 61,      "right_mouth": 291,
    "upper_lip": 13,       "lower_lip": 14,
    "chin": 18,            "forehead": 10,
    "left_cheek": 234,     "right_cheek": 454,
    "left_brow_outer": 70, "right_brow_outer": 300,
}

_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5,
)


def _dist(lm, a, b, w, h):
    pa = np.array([lm[a].x * w, lm[a].y * h])
    pb = np.array([lm[b].x * w, lm[b].y * h])
    return float(np.linalg.norm(pa - pb))


def geometric_features(img_np: np.ndarray) -> np.ndarray:
    H, W = img_np.shape[:2]
    rgb  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    res  = _mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return np.zeros(16, dtype=np.float32)
    lm = res.multi_face_landmarks[0].landmark
    L  = _LM
    face_h  = _dist(lm, L["forehead"],       L["chin"],           W, H) + 1e-6
    face_w  = _dist(lm, L["left_cheek"],     L["right_cheek"],    W, H) + 1e-6
    eye_d   = _dist(lm, L["left_eye_outer"], L["right_eye_outer"],W, H)
    mouth_w = _dist(lm, L["left_mouth"],     L["right_mouth"],    W, H)
    mouth_h = _dist(lm, L["upper_lip"],      L["lower_lip"],      W, H)
    l_eye_w = _dist(lm, L["left_eye_outer"], L["left_eye_inner"], W, H)
    r_eye_w = _dist(lm, L["right_eye_outer"],L["right_eye_inner"],W, H)
    brow_h_l= abs(lm[L["left_brow_outer"]].y  - lm[L["left_eye_outer"]].y)  * H
    brow_h_r= abs(lm[L["right_brow_outer"]].y - lm[L["right_eye_outer"]].y) * H
    chin_m  = _dist(lm, L["chin"],           L["lower_lip"],      W, H)
    nose_h  = _dist(lm, L["nose_bridge"],    L["nose_tip"],       W, H)
    nose_w  = mouth_w * 0.6
    left_x  = lm[L["left_cheek"]].x  * W
    right_x = lm[L["right_cheek"]].x * W
    symmetry= abs(lm[L["nose_tip"]].x * W - (left_x + right_x) / 2) / (face_w + 1e-6)
    feats = np.array([
        eye_d / face_h, l_eye_w / face_h, r_eye_w / face_h,
        nose_w / face_w, nose_h / face_h, mouth_w / face_w, mouth_h / face_h,
        face_w / face_h, brow_h_l / face_h, brow_h_r / face_h,
        chin_m / face_h,
        abs(l_eye_w - r_eye_w) / (l_eye_w + r_eye_w + 1e-6),
        abs(brow_h_l - brow_h_r) / (face_h + 1e-6),
        symmetry, nose_w / face_h, eye_d / face_w,
    ], dtype=np.float32)
    return np.clip(feats, 0.0, 5.0)


## ForensicCNN architecture — must match extract_features.py exactly so we can
## load the existing forensic_cnn.pth weights without errors.

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
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.head(feat), feat


def build_efficientnet() -> nn.Module:
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0, global_pool="avg")
    for p in model.parameters():
        p.requires_grad = False
    return model.eval().to(DEVICE)


def load_forensic_cnn() -> ForensicCNN:
    if not CNN_PATH.exists():
        log.error(f"forensic_cnn.pth not found at {CNN_PATH}")
        log.error("Run model_a/extract_features.py first to train the CNN.")
        raise SystemExit(1)
    model = ForensicCNN().to(DEVICE)
    model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    return model.eval()


@torch.no_grad()
def extract_video_features(
    frame_dir: Path,
    efficientnet: nn.Module,
    forensic_cnn: ForensicCNN,
) -> np.ndarray:
    jpgs = sorted(frame_dir.glob("*.jpg"))
    if not jpgs:
        return np.zeros(2064, dtype=np.float32)

    s1_vecs, s2_vecs, s3_vecs = [], [], []
    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t      = IMAGENET_NORM(TO_TENSOR(Image.fromarray(rgb))).unsqueeze(0).to(DEVICE)
        s1_vecs.append(efficientnet(t).squeeze().cpu().numpy())
        ft = frame_to_forensic_tensor(img).unsqueeze(0).to(DEVICE)
        _, s2 = forensic_cnn(ft)
        s2_vecs.append(s2.squeeze().cpu().numpy())
        s3_vecs.append(geometric_features(img))

    if not s1_vecs:
        return np.zeros(2064, dtype=np.float32)

    return np.concatenate([
        np.mean(s1_vecs, axis=0),
        np.mean(s2_vecs, axis=0),
        np.mean(s3_vecs, axis=0),
    ]).astype(np.float32)


def load_npz(path: Path) -> dict:
    """Load an NPZ and return it as a plain dict with mutable lists."""
    if not path.exists():
        log.error(f"NPZ not found: {path}")
        log.error("Run model_a/extract_features.py first.")
        raise SystemExit(1)
    d = np.load(path, allow_pickle=True)
    return {
        "X":         list(d["X"]),
        "y":         list(d["y"]),
        "video_ids": list(d["video_ids"]),
        "sources":   list(d["sources"]),
    }


def save_npz(path: Path, d: dict):
    np.savez(
        path,
        X         = np.array(d["X"],        dtype=np.float32),
        y         = np.array(d["y"],         dtype=np.int32),
        video_ids = np.array(d["video_ids"], dtype=object),
        sources   = np.array(d["sources"],   dtype=object),
    )


def load_checkpoint() -> set[str]:
    if CKPT_PATH.exists():
        return set(json.loads(CKPT_PATH.read_text()))
    return set()


def save_checkpoint(done: set[str]):
    CKPT_PATH.write_text(json.dumps(list(done)))


def main():
    log.info("=" * 62)
    log.info("Model A — Append Real Features (youtube_interviews)")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 62)
    log.info("")

    ## read only the youtube_interviews rows from manifest
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    new_rows = [r for r in all_rows if r["dataset_source"] == TARGET_SOURCE]
    if not new_rows:
        log.info("No youtube_interviews rows found in manifest.csv")
        log.info("Run collect_youtube_real.py first.")
        return

    rows_train = [r for r in new_rows if r["split"] == "train"]
    rows_test  = [r for r in new_rows if r["split"] == "test"]
    log.info(f"youtube_interviews videos — train: {len(rows_train)}  test: {len(rows_test)}")
    log.info("")

    ## load existing NPZs so we know which video_ids are already processed
    log.info("Loading existing NPZ files ...")
    data_train = load_npz(TRAIN_NPZ)
    data_test  = load_npz(TEST_NPZ)

    existing_ids = set(data_train["video_ids"]) | set(data_test["video_ids"])
    log.info(f"  Existing NPZ has {len(existing_ids)} videos total")

    ## load checkpoint (in case this append run was interrupted)
    done = load_checkpoint()
    log.info(f"  Already appended in a previous run: {len(done)} videos")
    log.info("")

    ## filter to only videos not yet in any NPZ
    to_process = [r for r in new_rows if r["video_id"] not in existing_ids and r["video_id"] not in done]
    if not to_process:
        log.info("All youtube_interviews videos are already in the NPZs — nothing to do.")
        CKPT_PATH.unlink(missing_ok=True)
        return

    log.info(f"Videos to process: {len(to_process)}")
    log.info("")

    ## load models (no CNN training — reuse the existing weights)
    log.info("Loading EfficientNet-B4 ...")
    efficientnet = build_efficientnet()

    log.info(f"Loading ForensicCNN from {CNN_PATH.name} ...")
    forensic_cnn = load_forensic_cnn()
    log.info("")

    ## process new videos and accumulate into the NPZ dicts
    new_train: dict[str, list] = {"X": [], "y": [], "video_ids": [], "sources": []}
    new_test:  dict[str, list] = {"X": [], "y": [], "video_ids": [], "sources": []}

    t_start = time.time()
    for i, row in enumerate(tqdm(to_process, desc="Extracting features"), start=1):
        frame_dir = ROOT / row["frame_dir"]
        feat      = extract_video_features(frame_dir, efficientnet, forensic_cnn)
        label     = LABEL_MAP[row["label"]]
        split     = row["split"]
        bucket    = new_train if split == "train" else new_test

        bucket["X"].append(feat)
        bucket["y"].append(label)
        bucket["video_ids"].append(row["video_id"])
        bucket["sources"].append(row["dataset_source"])

        done.add(row["video_id"])

        ## checkpoint every 100 videos
        if i % 100 == 0:
            save_checkpoint(done)
            elapsed = time.time() - t_start
            eta     = (len(to_process) - i) * (elapsed / i)
            log.info(f"  {i}/{len(to_process)}  elapsed={elapsed/60:.1f}min  ETA~{eta/60:.0f}min")

        if i % PAUSE_EVERY == 0:
            log.info(f"  [pause] cooling down for {PAUSE_SECS}s ...")
            time.sleep(PAUSE_SECS)

    ## append new rows to existing NPZs and save
    log.info("")
    log.info("Appending to NPZ files ...")

    for data_existing, new_data, split_name, path in [
        (data_train, new_train, "train", TRAIN_NPZ),
        (data_test,  new_test,  "test",  TEST_NPZ),
    ]:
        if not new_data["X"]:
            log.info(f"  {split_name}: no new rows to append")
            continue

        for key in ("X", "y", "video_ids", "sources"):
            data_existing[key].extend(new_data[key])

        save_npz(path, data_existing)
        total = len(data_existing["X"])
        added = len(new_data["X"])
        log.info(f"  {split_name}: added {added} rows  (total now: {total})  -> {path.name}")

    CKPT_PATH.unlink(missing_ok=True)
    log.info("")
    log.info("=" * 62)
    log.info("Done. NPZs updated with youtube_interviews features.")
    log.info("=" * 62)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Delete model_a/.sm1_checkpoint.json  (forces Submodel 1 to retrain)")
    log.info("  2. uv run model_a/train.py              (Submodel 2 skipped, Submodel 1 retrains)")


if __name__ == "__main__":
    main()
