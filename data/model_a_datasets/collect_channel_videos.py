## Model A — Channel-Specific Video Collection
##
## Downloads 60-second clips from a random position in each video,
## from specific YouTube channels known to be either real (human) or AI-generated.
##
## Real clips  → new_videos/real/<video_id>.mp4
##               frames saved to frames/<split>/real/youtube_channels/<video_id>/
##               Requires face detection in at least MIN_FACE_FRAMES early frames.
##
## AI clips    → new_videos/ai/<video_id>.mp4
##               frames saved to frames/<split>/ai_generated/youtube_channels/<video_id>/
##               No face check (AI videos don't need face presence).
##
## Resumable: checkpoint saved after every video.
## Re-run the script after any interruption — it skips already-processed videos.
##
## Run with:  uv run data/model_a_datasets/collect_channel_videos.py

import csv
import json
import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent.parent
DATA     = ROOT / "data" / "model_a_datasets"
NEW_VIDS = DATA / "new_videos"
FRAMES   = DATA / "frames"
MANIFEST = FRAMES / "manifest.csv"
CKPT     = DATA / ".channel_collect_checkpoint.json"

REAL_VID_DIR = NEW_VIDS / "real"
AI_VID_DIR   = NEW_VIDS / "ai"

REAL_CLIP_SECONDS = 60
AI_CLIP_SECONDS   = 80    ## longer clip for AI-generated to aid balance
IMG_SIZE          = 224
JPEG_QUALITY      = 95
VIDEOS_PER_CHANNEL = 50   ## fetch more videos per channel for better coverage

## face check for real videos
FACE_CHECK_FRAMES = 8
MIN_FACE_FRAMES   = 2

DATASET_SOURCE = "youtube_channels"

MANIFEST_FIELDS = [
    "video_id",
    "dataset_source",
    "label",
    "split",
    "frame_count",
    "frame_dir",
    "original_video_path",
]

## ── Channel lists ──────────────────────────────────────────────────────────

REAL_CHANNELS = [
    "https://www.youtube.com/@TheDiaryOfACEO",
    "https://www.youtube.com/@PeterMcKinnon",
    "https://www.youtube.com/@casey",
    "https://www.youtube.com/@YesTheory",
    "https://www.youtube.com/@lexfridman",
    "https://www.youtube.com/@KraigAdams",
    "https://www.youtube.com/@kold",
    "https://www.youtube.com/@SoftWhiteUnderbelly",
    "https://www.youtube.com/@johnnyharris",
    "https://www.youtube.com/@MattiHaapoja",
]

AI_CHANNELS = [
    "https://www.youtube.com/@VihannahArt",
    "https://www.youtube.com/@DanKieft",
    "https://www.youtube.com/@CuriousRefuge",
    "https://www.youtube.com/@AbandonedFilms",
    "https://www.youtube.com/@ThePromptist",
    "https://www.youtube.com/@AIVideoSchool",
    "https://www.youtube.com/@ReligiousStoryTV",
    "https://www.youtube.com/@ChronosHistory",
]

## Specific known AI-generated video IDs (from channel research)
AI_SEED_VIDEO_IDS = [
    "n0wolytOEac",  # Surreal Noir
    "mb0EAABT1XI",  # Mood Loops
    "FGIzsBOEyxI",  # Vihannah
    "bKpbqw_fySE",  # Dan Kieft
    "YY-M5bxTY28",  # AI Video School
]



def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def assign_split(accepted_count: int) -> str:
    return "test" if (accepted_count % 5 == 0) else "train"


def get_channel_video_ids(channel_url: str, max_videos: int) -> list[str]:
    """Fetch the most recent video IDs from a channel using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "id",
        "--playlist-end", str(max_videos),
        "--no-warnings",
        "--quiet",
        f"{channel_url}/videos",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        log.info(f"    {channel_url.split('@')[-1]}: {len(ids)} video IDs fetched")
        return ids
    except Exception as e:
        log.warning(f"    Failed to fetch channel {channel_url}: {e}")
        return []


def get_video_duration(video_id: str) -> int | None:
    """Return video duration in seconds without downloading."""
    cmd = [
        "yt-dlp",
        "--print", "%(duration)s",
        "--no-warnings",
        "--quiet",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration_str = result.stdout.strip()
        return int(float(duration_str)) if duration_str else None
    except Exception:
        return None


def download_segment(video_id: str, start: int, end: int, out_path: Path) -> bool:
    """Download a specific time range of a video using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "-f", "worst[ext=mp4]/worst",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000
    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning(f"    Download error for {video_id}: {e}")
        return False


def count_faces_in_frames(cap: cv2.VideoCapture, face_cascade: cv2.CascadeClassifier,
                           fps: float, total_frames: int) -> int:
    n_with_face = 0
    for s in range(min(FACE_CHECK_FRAMES, int(total_frames / max(fps, 1)))):
        frame_idx = int(s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            n_with_face += 1
    return n_with_face


def extract_frames(video_path: Path, out_dir: Path,
                   face_cascade: cv2.CascadeClassifier | None,
                   clip_seconds: int) -> int:
    """Extract 1-fps frames as 224×224 JPEGs.
    If face_cascade is provided, rejects clips with fewer than MIN_FACE_FRAMES faces.
    Returns frame count, -1 for face-check failure, 0 for unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    try:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / fps

        if face_cascade is not None:
            if count_faces_in_frames(cap, face_cascade, fps, total_frames) < MIN_FACE_FRAMES:
                return -1

        n_seconds      = min(int(duration_s), clip_seconds)
        sample_indices = [int(s * fps) for s in range(n_seconds) if int(s * fps) < total_frames]
        if not sample_indices:
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for n, frame_idx in enumerate(sample_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"frame_{n:04d}.jpg"), resized,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1
        return saved
    finally:
        cap.release()


def load_checkpoint() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text())
    return {"done": [], "real_accepted": 0, "ai_accepted": 0,
            "rejected_no_face": 0, "rejected_download": 0,
            "real_ids": [], "ai_ids": []}


def save_checkpoint(state: dict):
    CKPT.write_text(json.dumps(state, indent=2))



def collect_all_video_ids() -> tuple[list[str], list[str]]:
    """Fetch video IDs from all real and AI channels."""
    real_ids: list[str] = []
    ai_ids: list[str] = list(AI_SEED_VIDEO_IDS)  ## start with known seeds

    log.info("Fetching real channel video IDs ...")
    for channel in REAL_CHANNELS:
        ids = get_channel_video_ids(channel, VIDEOS_PER_CHANNEL)
        real_ids.extend(ids)
        time.sleep(1)

    log.info("Fetching AI channel video IDs ...")
    for channel in AI_CHANNELS:
        ids = get_channel_video_ids(channel, VIDEOS_PER_CHANNEL)
        ai_ids.extend(ids)
        time.sleep(1)

    ## deduplicate while preserving order
    real_ids = list(dict.fromkeys(real_ids))
    ai_ids   = list(dict.fromkeys(ai_ids))

    log.info(f"Total candidates — real: {len(real_ids)}, AI: {len(ai_ids)}")
    return real_ids, ai_ids


def process_video(video_id: str, label: str, vid_dir: Path,
                  face_cascade: cv2.CascadeClassifier | None,
                  n_accepted: int, clip_seconds: int) -> tuple[int, str | None]:
    """Download, clip, and extract frames for one video.
    Returns (frame_count, frame_dir_rel) or (negative, None) on failure."""

    ## get duration first — no downloading yet
    duration = get_video_duration(video_id)
    if duration is None or duration < clip_seconds + 5:
        log.info(f"    {video_id}: too short or duration unavailable, skipping")
        return 0, None

    ## pick a random start anywhere in the video except the last clip_seconds
    max_start = duration - clip_seconds
    start = random.randint(0, max_start)
    end   = start + clip_seconds

    clip_path = vid_dir / f"{video_id}.mp4"
    vid_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"    {video_id}: duration={duration}s, clip={start}s–{end}s")

    if not download_segment(video_id, start, end, clip_path):
        log.warning(f"    {video_id}: download failed")
        return -2, None  ## -2 = download failure

    split     = assign_split(n_accepted)
    frame_dir = FRAMES / split / label / DATASET_SOURCE / video_id

    frame_count = extract_frames(clip_path, frame_dir, face_cascade, clip_seconds)

    if frame_count == -1:
        log.info(f"    {video_id}: no face detected, skipping (real check)")
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        clip_path.unlink(missing_ok=True)
        return -1, None
    elif frame_count == 0:
        log.warning(f"    {video_id}: unreadable video")
        clip_path.unlink(missing_ok=True)
        return 0, None

    clip_path.unlink(missing_ok=True)
    log.info(f"    {video_id}: accepted [{split}] {frame_count} frames  "
             f"(clip={clip_seconds}s, video deleted)")
    return frame_count, _rel(frame_dir)


def main():
    REAL_VID_DIR.mkdir(parents=True, exist_ok=True)
    AI_VID_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    log.info("=" * 62)
    log.info("Model A — Channel-Specific Video Collection")
    log.info(f"Real clip   : {REAL_CLIP_SECONDS}s from a random position")
    log.info(f"AI clip     : {AI_CLIP_SECONDS}s from a random position")
    log.info(f"Per channel : {VIDEOS_PER_CHANNEL} videos")
    log.info("=" * 62)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        log.error("Could not load Haar cascade.")
        raise SystemExit(1)

    state      = load_checkpoint()
    done_ids   = set(state["done"])
    n_real     = state["real_accepted"]
    n_ai       = state["ai_accepted"]
    n_no_face  = state["rejected_no_face"]
    n_failed   = state["rejected_download"]

    if done_ids:
        log.info(f"Resuming — {n_real} real + {n_ai} AI already accepted, "
                 f"{len(done_ids)} tried.")

    if state.get("real_ids") and state.get("ai_ids"):
        real_ids = state["real_ids"]
        ai_ids   = state["ai_ids"]
        log.info(f"Using cached IDs — real: {len(real_ids)}, AI: {len(ai_ids)}")
    else:
        real_ids, ai_ids = collect_all_video_ids()
        state["real_ids"] = real_ids
        state["ai_ids"]   = ai_ids
        save_checkpoint(state)

    manifest_exists = MANIFEST.exists()
    manifest_file   = open(MANIFEST, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    if not manifest_exists:
        writer.writeheader()

    try:
        ## ── Real videos ──────────────────────────────────────────────────
        log.info("")
        log.info("Processing REAL videos ...")
        for video_id in real_ids:
            if video_id in done_ids:
                continue

            log.info(f"  [real {n_real}]  {video_id}")
            frame_count, frame_dir_rel = process_video(
                video_id, "real", REAL_VID_DIR, face_cascade, n_real,
                REAL_CLIP_SECONDS)

            if frame_count > 0 and frame_dir_rel:
                writer.writerow({
                    "video_id":            video_id,
                    "dataset_source":      DATASET_SOURCE,
                    "label":               "real",
                    "split":               assign_split(n_real),
                    "frame_count":         frame_count,
                    "frame_dir":           frame_dir_rel,
                    "original_video_path": f"new_videos/real/{video_id}.mp4",
                })
                manifest_file.flush()
                n_real += 1
            elif frame_count == -1:
                n_no_face += 1
            elif frame_count in (-2, 0):
                n_failed += 1

            done_ids.add(video_id)
            state.update({"done": list(done_ids), "real_accepted": n_real,
                          "ai_accepted": n_ai, "rejected_no_face": n_no_face,
                          "rejected_download": n_failed})
            save_checkpoint(state)
            time.sleep(0.5)

        ## ── AI-generated videos ──────────────────────────────────────────
        log.info("")
        log.info("Processing AI-GENERATED videos ...")

        def process_ai_batch(video_list: list[str]) -> None:
            nonlocal n_ai, n_failed
            for video_id in video_list:
                if video_id in done_ids:
                    continue

                log.info(f"  [ai {n_ai}]  {video_id}")
                frame_count, frame_dir_rel = process_video(
                    video_id, "ai_generated", AI_VID_DIR,
                    None,  ## no face check for AI videos
                    n_ai, AI_CLIP_SECONDS)

                if frame_count > 0 and frame_dir_rel:
                    writer.writerow({
                        "video_id":            video_id,
                        "dataset_source":      DATASET_SOURCE,
                        "label":               "ai_generated",
                        "split":               assign_split(n_ai),
                        "frame_count":         frame_count,
                        "frame_dir":           frame_dir_rel,
                        "original_video_path": f"new_videos/ai/{video_id}.mp4",
                    })
                    manifest_file.flush()
                    n_ai += 1
                elif frame_count in (-2, 0):
                    n_failed += 1

                done_ids.add(video_id)
                state.update({"done": list(done_ids), "real_accepted": n_real,
                              "ai_accepted": n_ai, "rejected_no_face": n_no_face,
                              "rejected_download": n_failed})
                save_checkpoint(state)
                time.sleep(0.5)

        process_ai_batch(ai_ids)

        ## ── Auto-balance: if real > AI, fetch more AI videos ─────────────
        if n_real > n_ai:
            deficit = n_real - n_ai
            log.info("")
            log.info(f"Balance check: real={n_real}, AI={n_ai} — "
                     f"fetching {deficit * 3} more AI candidates to close gap ...")

            extra_ai_ids: list[str] = []
            for channel in AI_CHANNELS:
                extra = get_channel_video_ids(channel, VIDEOS_PER_CHANNEL * 2)
                extra_ai_ids.extend(extra)
                time.sleep(1)
            extra_ai_ids = [v for v in dict.fromkeys(extra_ai_ids) if v not in done_ids]
            log.info(f"  {len(extra_ai_ids)} new AI candidates available")
            state["ai_ids"] = ai_ids + extra_ai_ids
            save_checkpoint(state)
            process_ai_batch(extra_ai_ids)

            if n_real > n_ai:
                log.info(f"After extra pass: real={n_real}, AI={n_ai} "
                         f"(gap reduced to {n_real - n_ai})")
            else:
                log.info(f"Balance achieved: real={n_real}, AI={n_ai}")

    finally:
        manifest_file.close()

    log.info("")
    log.info("=" * 62)
    log.info("Done.")
    log.info(f"  Real accepted       : {n_real}")
    log.info(f"  AI accepted         : {n_ai}")
    log.info(f"  Rejected (no face)  : {n_no_face}")
    log.info(f"  Rejected (download) : {n_failed}")
    log.info(f"  Manifest            : {_rel(MANIFEST)}")
    log.info("=" * 62)
    log.info("")
    log.info("Next steps:")
    log.info("  1. uv run model_a/append_real_features.py   (real frames)")
    log.info("  2. Retrain Submodel 1 with new real data")
    log.info("  3. Retrain Submodel 2 with new AI-gen data")


if __name__ == "__main__":
    main()
