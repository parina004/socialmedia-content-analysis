## For each of the 4,223 labelled videos this script:
##   1. Downloads the first 60 seconds using yt-dlp  (3 retries with backoff)
##   2. Extracts 20 features and logs every single one to the terminal
##   3. Writes the row to features.csv immediately (safe against crashes)
##   4. Deletes the clip right away — peak disk usage stays under ~500 MB

## Every video is wrapped in its own try/except so one bad video never
## kills the whole run. BATCH_SIZE controls how many videos per loop.
## After each batch it automatically starts the next one — no need to
## re-run manually. It stops only when all videos are done.

## yt-dlp uses browser cookies so YouTube doesn't block it as a bot.
## Make sure BROWSER below matches the browser you're logged into YouTube with.

## Run with:  uv run python data/extract_features.py

import csv
import gc
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import piq
import torch
from deepface import DeepFace
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


## file paths

BASE   = Path(__file__).parent
INPUT  = BASE / "labelled_videos.csv"
OUTPUT = BASE / "features.csv"
CKPT   = BASE / ".extract_checkpoint.json"
TMPDIR = BASE / ".tmp_clips"

TMPDIR.mkdir(exist_ok=True)


## settings

CLIP_SECONDS      = 60
FRAME_STEP_S      = 1      ## sample 1 frame per second
EMOTION_STEP      = 10     ## run DeepFace every 10 frames

BATCH_SIZE        = 200    ## how many videos to process per loop before pausing for GC
                           ## the script loops automatically — no need to re-run manually

DOWNLOAD_RETRIES  = 3      ## how many times to retry a failed download
GC_EVERY          = 50     ## call gc.collect() every N videos to release RAM

BROWSER           = "chrome"  ## browser to borrow YouTube cookies from — change to
                               ## "firefox", "safari", "edge", or "brave" if needed


## output columns

CSV_FIELDS = [
    "video_id", "youtube_url", "category", "label",
    "brisque_score", "color_vibrancy", "motion_intensity",
    "face_presence_ratio", "face_emotion_joy", "face_emotion_surprise",
    "thumbnail_brightness",
    "tempo_bpm", "rms_energy", "speech_ratio",
    "zero_crossing_rate", "beat_strength",
    "title_sentiment", "title_length", "title_has_question", "title_has_number",
    "description_length", "tag_count", "upload_hour", "upload_day",
    "like_to_view_ratio", "comment_to_view_ratio",
]


## checkpoint helpers

def load_checkpoint() -> tuple[set, set]:
    if CKPT.exists():
        data = json.loads(CKPT.read_text())
        return set(data.get("done", [])), set(data.get("failed", []))
    return set(), set()


def save_checkpoint(done: set, failed: set):
    CKPT.write_text(json.dumps({"done": list(done), "failed": list(failed)}, indent=2))


## misc helpers

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _cleanup_stale_files():
    ## deletes any .mp4 or .wav files left behind by a previous crash
    removed = 0
    for f in TMPDIR.glob("*"):
        if f.suffix in (".mp4", ".wav"):
            f.unlink(missing_ok=True)
            removed += 1
    if removed:
        log.info(f"Cleaned up {removed} stale temp file(s) from {TMPDIR}")


## download

def download_clip(video_id: str) -> "Path | None":
    out_path = TMPDIR / f"{video_id}.mp4"

    url = f"https://www.youtube.com/watch?v={video_id}"

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        if out_path.exists():
            out_path.unlink(missing_ok=True)

        result = subprocess.run(
            [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "--cookies-from-browser", BROWSER,
                "--format",
                "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
                "/best[height<=720][ext=mp4]/best[height<=720]",
                "--download-sections", f"*0-{CLIP_SECONDS}",
                "--force-keyframes-at-cuts",
                "--merge-output-format", "mp4",
                "--output", str(out_path),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        stderr = result.stderr.strip()[:300]

        ## permanent failures — no point retrying these
        if any(phrase in stderr for phrase in (
            "Video unavailable", "Private video", "has been removed",
            "This video is not available", "age-restricted",
        )):
            log.warning(f"  [download] permanent failure: {stderr[:120]}")
            return None

        if attempt < DOWNLOAD_RETRIES:
            wait = 2 ** attempt  ## 2s, then 4s
            log.warning(
                f"  [download] attempt {attempt}/{DOWNLOAD_RETRIES} failed — "
                f"retrying in {wait}s  ({stderr[:80]})"
            )
            time.sleep(wait)
        else:
            log.warning(f"  [download] all {DOWNLOAD_RETRIES} attempts failed — skipping")

    return None


## visual features

_face_detector = mp.solutions.face_detection.FaceDetection(
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

            if i % EMOTION_STEP == 0:
                try:
                    analysis = DeepFace.analyze(
                        img_path=rgb,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True,
                    )
                    emo = analysis[0]["emotion"] if isinstance(analysis, list) else analysis["emotion"]
                    joy_scores.append(emo.get("happy", 0.0))
                    surprise_scores.append(emo.get("surprise", 0.0))
                except Exception:
                    pass

    finally:
        cap.release()

    n = len(sample_indices) or 1
    result = {
        "brisque_score":         round(float(np.mean(brisque_scores)),    4) if brisque_scores   else float("nan"),
        "color_vibrancy":        round(float(np.mean(saturations)),        4) if saturations      else float("nan"),
        "motion_intensity":      round(float(np.mean(motion_values)),      4) if motion_values    else 0.0,
        "face_presence_ratio":   round(frames_with_face / n,               4),
        "face_emotion_joy":      round(float(np.mean(joy_scores)),         4) if joy_scores       else 0.0,
        "face_emotion_surprise": round(float(np.mean(surprise_scores)),    4) if surprise_scores  else 0.0,
        "thumbnail_brightness":  round(thumbnail_bright,                   4),
    }

    log.info(f"    [visual] brisque_score        = {result['brisque_score']}")
    log.info(f"    [visual] color_vibrancy        = {result['color_vibrancy']}")
    log.info(f"    [visual] motion_intensity       = {result['motion_intensity']}")
    log.info(f"    [visual] face_presence_ratio    = {result['face_presence_ratio']}")
    log.info(f"    [visual] face_emotion_joy       = {result['face_emotion_joy']}")
    log.info(f"    [visual] face_emotion_surprise  = {result['face_emotion_surprise']}")
    log.info(f"    [visual] thumbnail_brightness   = {result['thumbnail_brightness']}")

    return result


## audio features

def _extract_audio_track(video_path: Path) -> "Path | None":
    wav_path = TMPDIR / (video_path.stem + ".wav")
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
        log.warning("    [audio] ffmpeg failed — all audio features = nan")
        _log_audio(defaults)
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
            log.warning("    [audio] empty audio array — video may have no sound")
            _log_audio(defaults)
            return defaults

        log.info("    [audio] audio loaded via soundfile (numpy/scipy)")

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
        except Exception as e:
            log.warning(f"    [audio] tempo/beat_strength failed: {e}")

        result = {
            "tempo_bpm":          tempo_bpm,
            "rms_energy":         rms_energy,
            "speech_ratio":       speech_ratio,
            "zero_crossing_rate": zcr_mean,
            "beat_strength":      beat_strength,
        }
        _log_audio(result)
        return result

    except Exception as e:
        log.warning(f"    [audio] unexpected error — {e}")
        _log_audio(defaults)
        return defaults

    finally:
        if wav_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def _log_audio(f: dict):
    log.info(f"    [audio]  tempo_bpm            = {f['tempo_bpm']}")
    log.info(f"    [audio]  rms_energy            = {f['rms_energy']}")
    log.info(f"    [audio]  speech_ratio          = {f['speech_ratio']}")
    log.info(f"    [audio]  zero_crossing_rate    = {f['zero_crossing_rate']}")
    log.info(f"    [audio]  beat_strength         = {f['beat_strength']}")


## metadata features

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

    result = {
        "title_sentiment":    round(sentiment, 4),
        "title_length":       len(title),
        "title_has_question": has_question,
        "title_has_number":   has_number,
        "description_length": len(description),
        "tag_count":          tag_count,
        "upload_hour":        upload_hour,
        "upload_day":         upload_day,
    }

    log.info(f"    [meta]   title_sentiment       = {result['title_sentiment']}")
    log.info(f"    [meta]   title_length          = {result['title_length']}")
    log.info(f"    [meta]   title_has_question    = {result['title_has_question']}")
    log.info(f"    [meta]   title_has_number      = {result['title_has_number']}")
    log.info(f"    [meta]   description_length    = {result['description_length']}")
    log.info(f"    [meta]   tag_count             = {result['tag_count']}")
    log.info(f"    [meta]   upload_hour           = {result['upload_hour']}")
    log.info(f"    [meta]   upload_day            = {result['upload_day']}")

    return result


## main

def extract() -> bool:
    ## returns True if there are still videos left to process, False if all done
    if not INPUT.exists():
        raise FileNotFoundError("labelled_videos.csv not found — run label_videos.py first.")

    _cleanup_stale_files()

    log.info("Loading labelled_videos.csv ...")
    with open(INPUT, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    log.info(f"Loaded {len(all_rows)} videos")

    done, failed = load_checkpoint()
    log.info(f"Checkpoint: {len(done)} done, {len(failed)} unavailable")

    remaining = [r for r in all_rows if r["video_id"] not in done and r["video_id"] not in failed]
    log.info(f"Remaining : {len(remaining)} videos")

    if BATCH_SIZE > 0:
        batch = remaining[:BATCH_SIZE]
        log.info(f"Batch size: {BATCH_SIZE} — will stop after this batch and save checkpoint")
    else:
        batch = remaining
        log.info("Batch size: unlimited (running all remaining videos)")

    log.info("")

    file_exists = OUTPUT.exists()
    csv_file    = open(OUTPUT, "a", newline="", encoding="utf-8")
    writer      = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()

    total         = len(all_rows)
    n_done        = len(done)
    batch_success = 0
    batch_failed  = 0

    try:
        for row in batch:
            video_id    = row["video_id"]
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            n_done     += 1
            t_start     = time.time()

            log.info(f"{'─' * 70}")
            log.info(f"[{n_done}/{total}]  {video_id}")
            log.info(f"  category : {row['category']}")
            log.info(f"  label    : {row['label']}")
            log.info(f"  url      : {youtube_url}")
            log.info("")

            try:
                ## download the first 60 seconds
                log.info("  [step 1/4]  downloading first 60 seconds ...")
                clip_path = download_clip(video_id)
                if clip_path is None:
                    log.warning("  video unavailable — skipping")
                    failed.add(video_id)
                    save_checkpoint(done, failed)
                    batch_failed += 1
                    log.info("")
                    continue
                log.info(f"  download OK  ({clip_path.stat().st_size / 1_048_576:.1f} MB)")
                log.info("")

                ## extract visual and audio features, then delete the clip
                try:
                    log.info("  [step 2/4]  extracting visual features ...")
                    visual = extract_visual_features(clip_path)
                    log.info("")

                    log.info("  [step 3/4]  extracting audio features ...")
                    audio = extract_audio_features(clip_path)
                    log.info("")

                finally:
                    clip_path.unlink(missing_ok=True)
                    log.info("  clip deleted from disk")

                ## metadata comes straight from the CSV row, no video needed
                log.info("  [step 4/4]  extracting metadata features ...")
                metadata = extract_metadata_features(row)
                log.info("")

                ## ratios were pre-computed by label_videos.py
                ratios = {
                    "like_to_view_ratio":    _safe_float(row.get("like_to_view_ratio")),
                    "comment_to_view_ratio": _safe_float(row.get("comment_to_view_ratio")),
                }
                log.info(f"    [ratio]  like_to_view_ratio     = {ratios['like_to_view_ratio']}")
                log.info(f"    [ratio]  comment_to_view_ratio  = {ratios['comment_to_view_ratio']}")
                log.info("")

                ## write the completed row to features.csv
                out_row = {
                    "video_id":    video_id,
                    "youtube_url": youtube_url,
                    "category":    row["category"],
                    "label":       row["label"],
                    **visual, **audio, **metadata, **ratios,
                }
                writer.writerow(out_row)
                csv_file.flush()

                done.add(video_id)
                save_checkpoint(done, failed)
                batch_success += 1

                elapsed = time.time() - t_start
                log.info(
                    f"  row written  (total done: {len(done)})  "
                    f"[{elapsed:.1f}s]"
                )
                log.info("")

            except Exception as e:
                ## any unexpected error (OOM, codec crash, DeepFace failure, network issue)
                ## is caught here so the loop continues with the next video
                log.error(f"  ERROR processing {video_id}: {e}")
                log.error("  Skipping this video and continuing.")
                failed.add(video_id)
                save_checkpoint(done, failed)
                batch_failed += 1
                ## clean up any leftover clip file from the crash
                leftover = TMPDIR / f"{video_id}.mp4"
                leftover.unlink(missing_ok=True)
                log.info("")

            ## periodically free up RAM — prevents memory from growing across thousands of videos
            if (batch_success + batch_failed) % GC_EVERY == 0 and (batch_success + batch_failed) > 0:
                gc.collect()
                log.info(f"  [gc] collected — processed {batch_success + batch_failed} videos this run")
                log.info("")

    finally:
        csv_file.close()

    try:
        TMPDIR.rmdir()
    except OSError:
        pass  ## tmp dir not empty — leave it, cleanup runs at next startup

    remaining_count = len(all_rows) - len(done) - len(failed)

    log.info("=" * 70)
    log.info(f"Batch complete.")
    log.info(f"  this run  : {batch_success} extracted,  {batch_failed} skipped")
    log.info(f"  total done: {len(done)}")
    log.info(f"  remaining : {remaining_count}")
    log.info(f"  saved to  : {OUTPUT}")

    return remaining_count > 0


if __name__ == "__main__":
    batch_num = 1
    while True:
        log.info(f"{'=' * 70}")
        log.info(f"Starting batch {batch_num} ...")
        log.info(f"{'=' * 70}")
        has_more = extract()
        if not has_more:
            log.info("")
            log.info("All videos processed — extraction complete!")
            break
        log.info("")
        log.info(f"Batch {batch_num} done. Starting batch {batch_num + 1} automatically ...")
        log.info("")
        batch_num += 1
        gc.collect()
