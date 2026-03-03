## Model A — Feature Extraction (Step 1 of 2)

## Reads manifest.csv, runs every video's frames through 3 parallel streams,
## and saves per-video feature vectors as NPZ files for model_a/train.py.

## Stream 1  EfficientNet-B4 pretrained on ImageNet, backbone frozen.
##           GlobalAvgPool output = 1792-dim. Averaged across all frames.

## Stream 2  Forensic CNN trained from scratch on training frames.
##           Input = 6-channel tensor: 3 DCT channels (RGB) + 3 SRM noise
##           residual channels. GlobalAvgPool output = 256-dim.
##           Trained as 3-class classifier (real / deepfake / ai_generated)
##           then used as a feature extractor (head discarded).

## Stream 3  MediaPipe FaceMesh 468 landmarks -> 16 geometric ratios.
##           Averaged across frames. Zero vector when no face is detected.

## Final per-video vector: 1792 + 256 + 16 = 2064-dim.

## Outputs saved to model_a/:
##   forensic_cnn.pth       — trained Forensic CNN weights (Stream 2)
##   features_train.npz     — X (N,2064), y, video_ids, sources
##   features_test.npz      — same structure, held-out test split

## Checkpointed every video — safe to interrupt and resume.

## Run with:  uv run python model_a/extract_features.py

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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
MANIFEST = ROOT / "data" / "model_a_datasets" / "frames" / "manifest.csv"
OUT_DIR  = ROOT / "model_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_PATH   = OUT_DIR / "forensic_cnn.pth"
CKPT_PATH  = OUT_DIR / ".extract_checkpoint.json"
TRAIN_NPZ  = OUT_DIR / "features_train.npz"
TEST_NPZ   = OUT_DIR / "features_test.npz"

## label encoding (consistent across both submodels)
LABEL_MAP = {"real": 0, "deepfake": 1, "ai_generated": 2}

## training settings for the Forensic CNN
CNN_EPOCHS     = 10
CNN_BATCH      = 32
CNN_LR         = 1e-3
CNN_VAL_SPLIT  = 0.10   ## 10% of train frames held out for CNN validation

## pause for this many seconds every PAUSE_EVERY videos to let the CPU/MPS breathe
PAUSE_EVERY = 500
PAUSE_SECS  = 10

## device — prefer Apple MPS, then CUDA, then CPU
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)

## ImageNet normalisation for EfficientNet
IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
TO_TENSOR = transforms.ToTensor()


## SRM high-pass noise residual filters (3 key kernels from Fridrich 2012)
## These suppress image content and reveal noise patterns left by generators.
_SRM_K = [
    torch.tensor([[0, 0, 0], [ 0.5, -1,  0.5], [0, 0, 0]], dtype=torch.float32),  ## horiz
    torch.tensor([[0, 0.5, 0], [0, -1, 0], [0, 0.5, 0]],   dtype=torch.float32),  ## vert
    torch.tensor([[0.25, -0.5, 0.25], [-0.5, 1, -0.5], [0.25, -0.5, 0.25]], dtype=torch.float32),
]
## shape: (3, 1, 3, 3) — applied per grayscale channel via groups
SRM_KERNELS = torch.stack([k.unsqueeze(0).unsqueeze(0) for k in _SRM_K]).to(DEVICE)


def srm_residuals(gray_tensor: torch.Tensor) -> torch.Tensor:
    ## gray_tensor: (B, 1, H, W)  or  (1, H, W)
    if gray_tensor.dim() == 3:
        gray_tensor = gray_tensor.unsqueeze(0)
    out = []
    for k in SRM_KERNELS.unbind(0):
        r = F.conv2d(gray_tensor, k, padding=1)
        out.append(r)
    return torch.cat(out, dim=1)   ## (B, 3, H, W)


def compute_dct_channels(img_np: np.ndarray) -> np.ndarray:
    ## img_np: (H, W, 3) uint8  ->  returns (3, H, W) float32
    out = []
    for c in range(3):
        ch  = img_np[:, :, c].astype(np.float32) / 255.0
        dct = np.abs(scipy.fft.dctn(ch, norm="ortho"))
        dct = np.log1p(dct)
        lo, hi = dct.min(), dct.max()
        dct = (dct - lo) / (hi - lo + 1e-8)
        out.append(dct)
    return np.stack(out).astype(np.float32)   ## (3, H, W)


def frame_to_forensic_tensor(img_np: np.ndarray) -> torch.Tensor:
    ## img_np: (H, W, 3) BGR uint8  ->  (6, H, W) forensic tensor
    rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    dct = torch.tensor(compute_dct_channels(rgb))    ## (3, H, W)
    gray = torch.tensor(
        cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)                       ## (1, 1, H, W)
    srm = srm_residuals(gray.to(DEVICE)).squeeze(0).cpu()  ## (3, H, W)
    return torch.cat([dct, srm], dim=0)               ## (6, H, W)


## mediapipe key landmark indices
_LM = {
    "left_eye_outer":  33,  "right_eye_outer":  263,
    "left_eye_inner":  133, "right_eye_inner":  362,
    "nose_tip":        1,   "nose_bridge":      168,
    "left_mouth":      61,  "right_mouth":      291,
    "upper_lip":       13,  "lower_lip":        14,
    "chin":            18,  "forehead":         10,
    "left_cheek":      234, "right_cheek":      454,
    "left_brow_outer": 70,  "right_brow_outer": 300,
}

_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
)


def _dist(lm, a, b, w, h):
    pa = np.array([lm[a].x * w, lm[a].y * h])
    pb = np.array([lm[b].x * w, lm[b].y * h])
    return float(np.linalg.norm(pa - pb))


def geometric_features(img_np: np.ndarray) -> np.ndarray:
    ## img_np: (H, W, 3) BGR uint8  ->  (16,) float32, zeros if no face
    H, W = img_np.shape[:2]
    rgb  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    res  = _mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return np.zeros(16, dtype=np.float32)

    lm = res.multi_face_landmarks[0].landmark
    L  = _LM

    face_h = _dist(lm, L["forehead"],        L["chin"],            W, H) + 1e-6
    face_w = _dist(lm, L["left_cheek"],      L["right_cheek"],     W, H) + 1e-6
    eye_d  = _dist(lm, L["left_eye_outer"],  L["right_eye_outer"], W, H)
    nose_w = _dist(lm, L["left_mouth"],      L["right_mouth"],     W, H) * 0.6
    mouth_w= _dist(lm, L["left_mouth"],      L["right_mouth"],     W, H)
    mouth_h= _dist(lm, L["upper_lip"],       L["lower_lip"],       W, H)
    l_eye_w= _dist(lm, L["left_eye_outer"],  L["left_eye_inner"],  W, H)
    r_eye_w= _dist(lm, L["right_eye_outer"], L["right_eye_inner"], W, H)
    brow_h_l = abs(lm[L["left_brow_outer"]].y  - lm[L["left_eye_outer"]].y)  * H
    brow_h_r = abs(lm[L["right_brow_outer"]].y - lm[L["right_eye_outer"]].y) * H
    chin_mouth= _dist(lm, L["chin"],         L["lower_lip"],       W, H)
    nose_h = _dist(lm, L["nose_bridge"],     L["nose_tip"],        W, H)

    left_x  = lm[L["left_cheek"]].x  * W
    right_x = lm[L["right_cheek"]].x * W
    mid_x   = (left_x + right_x) / 2
    nose_x  = lm[L["nose_tip"]].x   * W
    symmetry = abs(nose_x - mid_x) / (face_w + 1e-6)

    feats = np.array([
        eye_d   / face_h,          ## 0  inter-eye dist ratio
        l_eye_w / face_h,          ## 1  left eye width ratio
        r_eye_w / face_h,          ## 2  right eye width ratio
        nose_w  / face_w,          ## 3  nose width ratio
        nose_h  / face_h,          ## 4  nose height ratio
        mouth_w / face_w,          ## 5  mouth width ratio
        mouth_h / face_h,          ## 6  mouth openness ratio
        face_w  / face_h,          ## 7  face aspect ratio
        brow_h_l / face_h,         ## 8  left brow height ratio
        brow_h_r / face_h,         ## 9  right brow height ratio
        chin_mouth / face_h,       ## 10 chin-to-mouth ratio
        abs(l_eye_w - r_eye_w) / (l_eye_w + r_eye_w + 1e-6),  ## 11 eye size symmetry
        abs(brow_h_l - brow_h_r) / (face_h + 1e-6),           ## 12 brow symmetry
        symmetry,                  ## 13 face horizontal symmetry
        nose_w / face_h,           ## 14 nose proportion
        eye_d / face_w,            ## 15 eye span ratio
    ], dtype=np.float32)

    return np.clip(feats, 0.0, 5.0)   ## clamp outliers


## Forensic CNN — definition and training logic

class ForensicCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.head(feat), feat   ## (logits, 256-dim features)


class ForensicFrameDataset(Dataset):
    def __init__(self, frame_paths: list[Path], labels: list[int]):
        self.paths  = frame_paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            return torch.zeros(6, 224, 224), self.labels[idx]
        return frame_to_forensic_tensor(img), self.labels[idx]


def train_forensic_cnn(rows_train: list[dict]) -> ForensicCNN:
    if CNN_PATH.exists():
        log.info("Forensic CNN weights found — loading cached model")
        model = ForensicCNN().to(DEVICE)
        model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
        model.eval()
        return model

    log.info("Training Forensic CNN from scratch ...")

    ## collect all frame paths + frame-level labels from training videos
    all_paths, all_labels = [], []
    for row in rows_train:
        frame_dir = ROOT / row["frame_dir"]
        label     = LABEL_MAP[row["label"]]
        for jpg in sorted(frame_dir.glob("*.jpg")):
            all_paths.append(jpg)
            all_labels.append(label)

    log.info(f"  {len(all_paths)} training frames across {len(rows_train)} videos")

    ## train / val split
    n_val   = max(1, int(len(all_paths) * CNN_VAL_SPLIT))
    idx     = np.random.permutation(len(all_paths))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    trn_ds = ForensicFrameDataset([all_paths[i] for i in trn_idx], [all_labels[i] for i in trn_idx])
    val_ds = ForensicFrameDataset([all_paths[i] for i in val_idx], [all_labels[i] for i in val_idx])

    trn_dl = DataLoader(trn_ds, batch_size=CNN_BATCH, shuffle=True,  num_workers=2, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False, num_workers=2, pin_memory=False)

    model     = ForensicCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        trn_loss = 0.0
        for x, y in tqdm(trn_dl, desc=f"  Epoch {epoch}/{CNN_EPOCHS} [train]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"  Epoch {epoch}/{CNN_EPOCHS} [val] ", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                val_loss += criterion(logits, y).item()
                correct  += (logits.argmax(1) == y).sum().item()

        avg_trn = trn_loss / len(trn_dl)
        avg_val = val_loss / len(val_dl)
        val_acc = correct / len(val_ds)
        log.info(f"  Epoch {epoch}/{CNN_EPOCHS}  train_loss={avg_trn:.4f}  val_loss={avg_val:.4f}  val_acc={val_acc:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), CNN_PATH)
            log.info(f"    -> saved best model (val_loss={avg_val:.4f})")

    log.info(f"Forensic CNN training complete. Best val_loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    model.eval()
    return model


## EfficientNet-B4 pretrained feature extractor (Stream 1)

def build_efficientnet() -> nn.Module:
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0, global_pool="avg")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(DEVICE)


## runs all 3 streams on every frame in a video folder and returns the averaged vector

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

        ## Stream 1 — EfficientNet
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil    = Image.fromarray(rgb)
        t      = IMAGENET_NORM(TO_TENSOR(pil)).unsqueeze(0).to(DEVICE)
        s1_vec = efficientnet(t).squeeze().cpu().numpy()   ## (1792,)
        s1_vecs.append(s1_vec)

        ## Stream 2 — Forensic CNN
        forensic_t = frame_to_forensic_tensor(img).unsqueeze(0).to(DEVICE)
        _, s2_vec  = forensic_cnn(forensic_t)
        s2_vecs.append(s2_vec.squeeze().cpu().numpy())     ## (256,)

        ## Stream 3 — MediaPipe geometric features
        s3_vecs.append(geometric_features(img))            ## (16,)

    if not s1_vecs:
        return np.zeros(2064, dtype=np.float32)

    s1 = np.mean(s1_vecs, axis=0)   ## (1792,)
    s2 = np.mean(s2_vecs, axis=0)   ## (256,)
    s3 = np.mean(s3_vecs, axis=0)   ## (16,)

    return np.concatenate([s1, s2, s3]).astype(np.float32)  ## (2064,)


## checkpoint helpers so a long run can be interrupted and resumed safely

def load_checkpoint() -> dict[str, np.ndarray]:
    if CKPT_PATH.exists():
        raw = json.loads(CKPT_PATH.read_text())
        return {k: np.array(v) for k, v in raw.items()}
    return {}


def save_checkpoint(done: dict[str, np.ndarray]):
    CKPT_PATH.write_text(json.dumps({k: v.tolist() for k, v in done.items()}))


def main():
    log.info("=" * 62)
    log.info("Model A — Feature Extraction")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 62)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows_train = [r for r in rows if r["split"] == "train"]
    rows_test  = [r for r in rows if r["split"] == "test"]
    log.info(f"Videos — train: {len(rows_train)}  test: {len(rows_test)}")
    log.info("")

    ## build models
    log.info("Loading EfficientNet-B4 (pretrained, frozen) ...")
    efficientnet = build_efficientnet()

    log.info("Preparing Forensic CNN (Stream 2) ...")
    forensic_cnn = train_forensic_cnn(rows_train)
    forensic_cnn.eval()

    ## load any previous checkpoint
    done = load_checkpoint()
    log.info(f"Resuming from checkpoint: {len(done)} videos already done")
    log.info("")

    results: dict[str, list] = {
        "train": {"X": [], "y": [], "video_ids": [], "sources": []},
        "test":  {"X": [], "y": [], "video_ids": [], "sources": []},
    }

    ## add already-done videos back into results (they're in the checkpoint)
    ## checkpoint stores video_id -> feature vector
    ## we need to match them back to their split/label for the NPZ
    id_to_row = {f"{r['dataset_source']}__{r['video_id']}": r for r in rows}
    for uid, feat in done.items():
        row = id_to_row.get(uid)
        if row is None:
            continue
        split = row["split"]
        results[split]["X"].append(feat)
        results[split]["y"].append(LABEL_MAP[row["label"]])
        results[split]["video_ids"].append(row["video_id"])
        results[split]["sources"].append(row["dataset_source"])

    t_start = time.time()
    n_total = len(rows)
    n_done  = len(done)

    for row in tqdm(rows, desc="Extracting features"):
        uid = f"{row['dataset_source']}__{row['video_id']}"
        if uid in done:
            continue

        frame_dir = ROOT / row["frame_dir"]
        feat = extract_video_features(frame_dir, efficientnet, forensic_cnn)

        split = row["split"]
        results[split]["X"].append(feat)
        results[split]["y"].append(LABEL_MAP[row["label"]])
        results[split]["video_ids"].append(row["video_id"])
        results[split]["sources"].append(row["dataset_source"])

        done[uid] = feat
        n_done += 1

        ## checkpoint every 100 videos
        if n_done % 100 == 0:
            save_checkpoint(done)
            elapsed = time.time() - t_start
            rate    = elapsed / max(n_done - len(results["train"]["X"]) - len(results["test"]["X"]) + len(done), 1)
            eta     = (n_total - n_done) * (elapsed / n_done)
            log.info(f"  {n_done}/{n_total}  elapsed={elapsed/60:.1f}min  ETA~{eta/60:.0f}min")

        ## every PAUSE_EVERY videos take a short break so the machine doesn't overheat
        if n_done % PAUSE_EVERY == 0:
            log.info(f"  [pause] cooling down for {PAUSE_SECS}s ...")
            time.sleep(PAUSE_SECS)

    ## save NPZ files
    for split, npz_path in [("train", TRAIN_NPZ), ("test", TEST_NPZ)]:
        d = results[split]
        np.savez(
            npz_path,
            X         = np.array(d["X"],        dtype=np.float32),
            y         = np.array(d["y"],         dtype=np.int32),
            video_ids = np.array(d["video_ids"], dtype=object),
            sources   = np.array(d["sources"],   dtype=object),
        )
        log.info(f"Saved {split}: {len(d['X'])} videos -> {npz_path.name}")

    ## clean up checkpoint
    CKPT_PATH.unlink(missing_ok=True)
    log.info("Checkpoint removed.")
    log.info("")
    log.info("=" * 62)
    log.info("Feature extraction complete. Run model_a/train.py next.")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
