## model_b/features_inference.py
## Lightweight feature extraction for inference (deployment).
## Identical to data/model_b_datasets/extract_features.py except:
##   - No DeepFace / TensorFlow import
##   - face_emotion_joy and face_emotion_surprise default to 0.0
## The original training script is untouched — use that locally for data collection.

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe.python.solutions.face_detection as _mp_face_det
import numpy as np
import piq
import torch
from textblob import TextBlob

log = logging.getLogger(__name__)

CLIP_SECONDS = 30
EMOTION_STEP = 3   # kept so frame-loop logic is identical; emotion block is just skipped

_face_detector = _mp_face_det.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5,
)


def _brisque(frame_rgb: np.ndarray) -> float:
    if frame_rgb.shape[0] < 32 or frame_rgb.shape[1] < 32:
        return float("nan")
    try:
        t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return float(piq.brisque(t, data_range=1.0))
    except Exception:
        return float("nan")


def extract_visual_features(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sample_indices = [
            int(i * fps)
            for i in range(CLIP_SECONDS)
            if int(i * fps) < total_frames
        ]

        brisque_scores   = []
        saturations      = []
        motion_values    = []
        frames_with_face = 0
        thumbnail_bright = float("nan")
        prev_gray        = None
        # DeepFace removed — emotion features default to 0.0 via empty-list fallback below
        joy_scores       = []
        surprise_scores  = []

        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if i == 0:
                gray_first       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thumbnail_bright = float(gray_first.mean())

            bq = _brisque(rgb)
            if not np.isnan(bq):
                brisque_scores.append(bq)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturations.append(float(hsv[:, :, 1].mean()))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                motion_values.append(
                    float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
                )
            prev_gray = gray

            if _face_detector.process(rgb).detections:
                frames_with_face += 1

    finally:
        cap.release()

    n = len(sample_indices) or 1
    return {
        "brisque_score":         round(float(np.mean(brisque_scores)),    4) if brisque_scores   else float("nan"),
        "color_vibrancy":        round(float(np.mean(saturations)),        4) if saturations      else float("nan"),
        "motion_intensity":      round(float(np.mean(motion_values)),      4) if motion_values    else 0.0,
        "face_presence_ratio":   round(frames_with_face / n,               4),
        "face_emotion_joy":      round(float(np.mean(joy_scores)),         4) if joy_scores       else 0.0,
        "face_emotion_surprise": round(float(np.mean(surprise_scores)),    4) if surprise_scores  else 0.0,
        "thumbnail_brightness":  round(thumbnail_bright,                   4),
    }


def _extract_audio_track(video_path: Path) -> "Path | None":
    tmp_dir  = video_path.parent
    wav_path = tmp_dir / (video_path.stem + "_inf.wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            "-t", str(CLIP_SECONDS),
            str(wav_path),
        ],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0 or not wav_path.exists():
        return None
    return wav_path


def extract_audio_features(video_path: Path) -> dict:
    nan = float("nan")
    defaults = {
        "tempo_bpm": nan, "rms_energy": nan, "speech_ratio": nan,
        "zero_crossing_rate": nan, "beat_strength": nan,
    }

    wav_path = _extract_audio_track(video_path)
    if wav_path is None:
        return defaults

    try:
        import soundfile as sf
        data, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        y = data.mean(axis=1) if data.ndim > 1 else data

        if samplerate != 22050:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(22050, int(samplerate))
            y = resample_poly(y, 22050 // g, int(samplerate) // g).astype(np.float32)

        if len(y) == 0:
            return defaults

        frame_length = 2048
        hop_length   = 512
        n_frames     = max(1, 1 + (len(y) - frame_length) // hop_length)

        rms_vals = np.zeros(n_frames, dtype=np.float32)
        zcr_vals = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            frame = y[i * hop_length : i * hop_length + frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            rms_vals[i] = np.sqrt(np.mean(frame ** 2))
            zcr_vals[i] = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)

        rms_energy   = round(float(np.mean(rms_vals)), 6)
        zcr_mean     = round(float(np.mean(zcr_vals)), 6)
        speech_ratio = round(float(np.mean((zcr_vals > 0.05) & (zcr_vals < 0.35))), 4)

        tempo_bpm     = nan
        beat_strength = nan
        try:
            from scipy.signal import correlate
            onset_env = np.maximum(0.0, np.diff(rms_vals, prepend=rms_vals[0]))
            if onset_env.std() > 1e-8:
                ac         = correlate(onset_env, onset_env, mode="full")[len(onset_env) - 1:]
                ac         = ac / (ac[0] + 1e-8)
                frame_rate = 22050 / hop_length
                p_min = max(2, int(frame_rate * 60 / 200))
                p_max = min(len(ac) - 1, int(frame_rate * 60 / 40))
                if p_max > p_min:
                    region        = ac[p_min:p_max]
                    best_offset   = int(np.argmax(region))
                    tempo_bpm     = round(float(frame_rate * 60 / (best_offset + p_min)), 2)
                    beat_strength = round(float(region[best_offset]), 6)
        except Exception:
            pass

        return {
            "tempo_bpm":          tempo_bpm,
            "rms_energy":         rms_energy,
            "speech_ratio":       speech_ratio,
            "zero_crossing_rate": zcr_mean,
            "beat_strength":      beat_strength,
        }

    except Exception:
        return defaults

    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def extract_metadata_features(row: dict) -> dict:
    title       = row.get("title", "")       or ""
    description = row.get("description", "") or ""
    tags_str    = row.get("tags", "")        or ""
    upload_date = row.get("upload_date", "") or ""

    sentiment    = float(TextBlob(title).sentiment.polarity)
    has_question = 1 if "?" in title else 0
    has_number   = 1 if re.search(r"\d", title) else 0
    tag_count    = len([t for t in tags_str.split("|") if t.strip()]) if tags_str else 0

    try:
        dt          = datetime.fromisoformat(upload_date.replace("Z", "+00:00"))
        upload_hour = dt.hour
        upload_day  = dt.weekday()
    except Exception:
        upload_hour = -1
        upload_day  = -1

    return {
        "title_sentiment":    round(sentiment, 4),
        "title_length":       len(title),
        "title_has_question": has_question,
        "title_has_number":   has_number,
        "description_length": len(description),
        "tag_count":          tag_count,
        "upload_hour":        upload_hour,
        "upload_day":         upload_day,
    }
