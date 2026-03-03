## Model A — Real Face Video Collection (YouTube API + yt-dlp)
##
## Searches YouTube for videos where real people are talking on camera
## (interviews, stand-up comedy, TED talks, podcasts, news, etc.),
## downloads only the first 30 seconds of each clip using yt-dlp,
## checks that a human face is visible in multiple frames,
## extracts frames at 1 fps, and appends results to manifest.csv.
##
## Batch mode: videos are processed in batches of BATCH_SIZE with a short
## sleep between batches so the system is not continuously hammered.
##
## Resumable: a checkpoint file is saved after every single video.
## If the script is interrupted, just re-run it — it will automatically
## pick up from where it left off without reprocessing anything.
##
## Only 30 seconds are downloaded per video, so total disk usage is
## roughly 2-5 MB per clip — about 4-10 GB for 2000 clips total.
##
## Run with:  uv run data/model_a_datasets/collect_youtube_real.py

import csv
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


## ── paths ────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent.parent.parent   ## project root
DATA     = ROOT / "data" / "model_a_datasets"
OUT      = DATA / "frames"
MANIFEST = OUT / "manifest.csv"
CKPT     = DATA / ".youtube_real_checkpoint.json"
TMPDIR   = DATA / ".tmp_youtube_real"


## ── collection settings ──────────────────────────────────────────────────────

TARGET_VIDEOS     = 2000   ## stop after this many accepted face clips
MAX_FRAMES        = 30     ## max frames per clip (1 fps = 30 seconds)
IMG_SIZE          = 224    ## resize all frames to 224×224
JPEG_QUALITY      = 95
CLIP_SECONDS      = 30     ## seconds to download per video

## face check — require a face in at least MIN_FACE_FRAMES out of the first
## FACE_CHECK_FRAMES sampled seconds, to reject videos with no people in them
FACE_CHECK_FRAMES = 8      ## how many early frames to check
MIN_FACE_FRAMES   = 2      ## minimum frames that must contain a face

## candidate pool — fetch 5× more IDs than needed to cover ~50% rejection rate
CANDIDATE_MULTIPLIER  = 5
MAX_RESULTS_PER_QUERY = 200   ## paginated: 4 pages × 50 per page

## batch settings — process BATCH_SIZE videos, sleep BATCH_SLEEP seconds,
## then continue automatically with the next batch
BATCH_SIZE  = 20    ## videos per batch before taking a break
BATCH_SLEEP = 10    ## seconds to sleep between batches

DATASET_SOURCE = "youtube_interviews"
LABEL          = "real"

MANIFEST_FIELDS = [
    "video_id",
    "dataset_source",
    "label",
    "split",
    "frame_count",
    "frame_dir",
    "original_video_path",
]


## ── search queries ───────────────────────────────────────────────────────────

## 30 queries chosen to maximise chance of real people talking on camera.
## 30 × up to 200 results = up to 6,000 candidates, which easily covers
## the 2,000 target after face filtering.
SEARCH_QUERIES = [
    "celebrity interview 2023",
    "stand up comedy show",
    "TED talk",
    "podcast interview",
    "news anchor interview",
    "late night show interview",
    "press conference",
    "motivational speaker",
    "documentary talking head",
    "political speech",
    "science lecture university",
    "comedy show performance",
    "journalist interview",
    "startup founder interview",
    "athlete interview",
    "actor interview",
    "author book interview",
    "CEO business interview",
    "doctor medical talk",
    "professor lecture",
    "comedian monologue",
    "news panel discussion",
    "debate speech",
    "TEDx talk",
    "talk show guest",
    "film director interview",
    "musician interview",
    "youtuber vlog talking",
    "expert opinion interview",
    "conference keynote speech",
]

## reuses the same API key env vars as Model B collection
API_KEYS = [k for k in [
    os.getenv("YT_API_1"),
    os.getenv("YT_API_2"),
    os.getenv("YT_API_3"),
] if k]


## ── helpers ──────────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    """Return a path relative to the project root, for storing in manifest.csv."""
    return str(path.relative_to(ROOT))


def build_youtube_client(key: str):
    return build("youtube", "v3", developerKey=key)


## ── YouTube search ───────────────────────────────────────────────────────────

def search_videos(youtube, query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[str]:
    """Search YouTube for a query and return video IDs.
    Paginates automatically until max_results or no more pages."""
    ids: list[str] = []
    next_page_token = None

    while len(ids) < max_results:
        page_size = min(50, max_results - len(ids))  ## YouTube cap is 50 per page
        try:
            response = youtube.search().list(
                q=query,
                type="video",
                part="id",
                maxResults=page_size,
                videoDuration="medium",   ## 4-20 min clips — proper talking-head content
                safeSearch="strict",
                relevanceLanguage="en",
                pageToken=next_page_token,
            ).execute()
        except HttpError as e:
            if e.resp.status == 403:
                raise  ## quota exhausted — caller will rotate to next API key
            log.warning(f"  Search error for '{query}': {e}")
            break

        page_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        ids.extend(page_ids)
        next_page_token = response.get("nextPageToken")

        if not next_page_token or not page_ids:
            break

        time.sleep(0.3)

    return ids


def collect_video_ids(target: int) -> list[str]:
    """Collect candidate video IDs across all queries and API keys.
    target is the inflated count (remaining_needed × CANDIDATE_MULTIPLIER)."""
    if not API_KEYS:
        log.error(
            "No YouTube API keys found. Set YT_API_1 / YT_API_2 / YT_API_3 in .env"
        )
        raise SystemExit(1)

    seen: set[str] = set()
    video_ids: list[str] = []
    key_index = 0

    youtube = build_youtube_client(API_KEYS[key_index])
    log.info(f"  Using API key #{key_index + 1} of {len(API_KEYS)}")

    for query in SEARCH_QUERIES:
        if len(video_ids) >= target:
            break
        log.info(f"  Searching: '{query}' ...")
        try:
            ids = search_videos(youtube, query)
        except HttpError as e:
            if e.resp.status == 403:
                log.warning(f"  API key #{key_index + 1} quota exhausted.")
                key_index += 1
                if key_index >= len(API_KEYS):
                    log.error("  All API keys exhausted. Stopping collection.")
                    break
                log.info(f"  Switching to API key #{key_index + 1}.")
                youtube = build_youtube_client(API_KEYS[key_index])
                ids = search_videos(youtube, query)  ## retry with new key, full pagination
            else:
                log.warning(f"  Skipping query due to error: {e}")
                ids = []

        new_ids = [vid for vid in ids if vid not in seen]
        seen.update(new_ids)
        video_ids.extend(new_ids)
        log.info(f"    -> {len(new_ids)} new IDs  (total: {len(video_ids)})")
        time.sleep(0.5)

    log.info(f"  Collected {len(video_ids)} candidate IDs total")
    return video_ids


## ── download ─────────────────────────────────────────────────────────────────

def download_clip(video_id: str, out_path: Path) -> bool:
    """Download the first CLIP_SECONDS seconds of a YouTube video.
    Uses yt-dlp at the lowest available quality — we only need visual features."""
    cmd = [
        "yt-dlp",
        "--download-sections", f"*0-{CLIP_SECONDS}",
        "-f", "worst[ext=mp4]/worst",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000
    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning(f"    Download error for {video_id}: {e}")
        return False


## ── face check + frame extraction ───────────────────────────────────────────

def count_faces_in_early_frames(
    cap: cv2.VideoCapture,
    face_cascade: cv2.CascadeClassifier,
    fps: float,
    total_frames: int,
) -> int:
    """Sample the first FACE_CHECK_FRAMES seconds and count how many contain a face.
    This is intentionally strict — we want real people clearly visible on camera."""
    n_with_face = 0
    for s in range(min(FACE_CHECK_FRAMES, int(total_frames / fps))):
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


def extract_frames_with_face_check(
    video_path: Path,
    out_dir: Path,
    face_cascade: cv2.CascadeClassifier,
) -> int:
    """Extract 1-fps frames (up to MAX_FRAMES), saving as 224×224 JPEGs.
    First verifies that at least MIN_FACE_FRAMES early frames contain a face.
    Returns number of frames saved, or -1 if the face check failed."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    try:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / fps

        ## face check first — reject quickly if there's no person visible
        n_face_frames = count_faces_in_early_frames(cap, face_cascade, fps, total_frames)
        if n_face_frames < MIN_FACE_FRAMES:
            return -1  ## -1 = no face (distinct from 0 = unreadable video)

        ## face confirmed — extract frames at 1 fps
        n_seconds      = min(int(duration_s), MAX_FRAMES)
        sample_indices = [
            int(s * fps)
            for s in range(n_seconds)
            if int(s * fps) < total_frames
        ]

        if not sample_indices:
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for n, frame_idx in enumerate(sample_indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            resized  = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"frame_{n:04d}.jpg"
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

        return saved

    finally:
        cap.release()


## ── checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load the checkpoint file so we can resume after an interruption.
    If no checkpoint exists, return a fresh state."""
    if CKPT.exists():
        return json.loads(CKPT.read_text())
    return {"done": [], "accepted": 0, "rejected_no_face": 0, "rejected_download": 0}


def save_checkpoint(state: dict):
    CKPT.write_text(json.dumps(state, indent=2))


def assign_split(accepted_count: int) -> str:
    """Every 5th accepted clip goes to test (20%), the rest to train (80%)."""
    return "test" if (accepted_count % 5 == 0) else "train"


## ── main ─────────────────────────────────────────────────────────────────────

def main():
    TMPDIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("=" * 62)
    log.info("Model A — Real Face Collection via YouTube")
    log.info(f"Target  : {TARGET_VIDEOS} clips with confirmed faces")
    log.info(f"Batch   : {BATCH_SIZE} videos, then {BATCH_SLEEP}s sleep")
    log.info("=" * 62)
    log.info("")

    ## load OpenCV Haar cascade — bundled with OpenCV, no extra download needed
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        log.error("Could not load Haar cascade. Check your OpenCV installation.")
        raise SystemExit(1)

    ## load checkpoint — if the script was interrupted, this picks up where it left off
    state = load_checkpoint()
    done_ids   = set(state["done"])
    n_accepted = state["accepted"]
    n_no_face  = state["rejected_no_face"]
    n_failed   = state["rejected_download"]

    if done_ids:
        log.info(f"Resuming — {n_accepted} clips already accepted, {len(done_ids)} tried.")
        log.info("")

    if n_accepted >= TARGET_VIDEOS:
        log.info(f"Already have {n_accepted} clips — target reached, nothing to do.")
        return

    ## collect candidate IDs — 5× more than we need to cover ~50% rejection rate
    remaining_needed = TARGET_VIDEOS - n_accepted
    log.info(f"Need {remaining_needed} more clips — fetching {remaining_needed * CANDIDATE_MULTIPLIER} candidates ...")
    log.info("")
    candidate_ids = collect_video_ids(target=remaining_needed * CANDIDATE_MULTIPLIER)
    log.info("")

    new_candidates = [vid for vid in candidate_ids if vid not in done_ids]
    log.info(f"New candidates to process: {len(new_candidates)}")
    log.info("")

    if not new_candidates:
        log.info("No new candidates. Try again tomorrow when API quota resets.")
        return

    ## open manifest in append mode — existing rows are never touched
    manifest_exists = MANIFEST.exists()
    manifest_file   = open(MANIFEST, "a", newline="", encoding="utf-8")
    writer          = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    if not manifest_exists:
        writer.writeheader()

    log.info("Starting batch download + frame extraction ...")
    log.info(f"(Batches of {BATCH_SIZE} with a {BATCH_SLEEP}s break in between)")
    log.info("")

    try:
        batch_count = 0  ## videos processed in the current batch

        for video_id in new_candidates:
            if n_accepted >= TARGET_VIDEOS:
                break

            ## take a break after every BATCH_SIZE videos to rest the system
            if batch_count > 0 and batch_count % BATCH_SIZE == 0:
                log.info(f"  -- batch of {BATCH_SIZE} done, sleeping {BATCH_SLEEP}s --")
                time.sleep(BATCH_SLEEP)

            tmp_path = TMPDIR / f"{video_id}.mp4"

            log.info(f"  [{n_accepted}/{TARGET_VIDEOS}]  {video_id}")

            ## step 1: download the first 30 seconds
            success = download_clip(video_id, tmp_path)
            if not success:
                log.warning(f"    -> download failed")
                n_failed += 1
                done_ids.add(video_id)
                state.update({"done": list(done_ids), "accepted": n_accepted,
                               "rejected_no_face": n_no_face, "rejected_download": n_failed})
                save_checkpoint(state)
                batch_count += 1
                continue

            ## step 2: face check + frame extraction
            split     = assign_split(n_accepted)
            frame_dir = OUT / split / LABEL / DATASET_SOURCE / video_id
            frame_count = extract_frames_with_face_check(tmp_path, frame_dir, face_cascade)

            ## clean up the temp clip right away — no reason to keep it
            tmp_path.unlink(missing_ok=True)

            if frame_count == -1:
                log.info(f"    -> no face detected, skipping")
                n_no_face += 1
                if frame_dir.exists():
                    shutil.rmtree(frame_dir)
            elif frame_count == 0:
                log.warning(f"    -> unreadable video, skipping")
                n_failed += 1
            else:
                ## accepted — add to manifest
                writer.writerow({
                    "video_id":            video_id,
                    "dataset_source":      DATASET_SOURCE,
                    "label":               LABEL,
                    "split":               split,
                    "frame_count":         frame_count,
                    "frame_dir":           _rel(frame_dir),
                    "original_video_path": f"youtube/{video_id}",
                })
                manifest_file.flush()
                n_accepted += 1
                log.info(f"    -> accepted [{split}]  {frame_count} frames  (total: {n_accepted})")

            ## save checkpoint after every video so interruptions are safe
            done_ids.add(video_id)
            state.update({"done": list(done_ids), "accepted": n_accepted,
                           "rejected_no_face": n_no_face, "rejected_download": n_failed})
            save_checkpoint(state)
            batch_count += 1

    finally:
        manifest_file.close()

    ## remove temp dir if it's empty
    try:
        TMPDIR.rmdir()
    except OSError:
        pass

    if n_accepted >= TARGET_VIDEOS:
        CKPT.unlink(missing_ok=True)
        log.info("Checkpoint removed — collection complete.")

    log.info("")
    log.info("=" * 62)
    log.info("Done.")
    log.info(f"  Accepted (face confirmed) : {n_accepted}")
    log.info(f"  Rejected (no face)        : {n_no_face}")
    log.info(f"  Rejected (download fail)  : {n_failed}")
    log.info(f"  Manifest                  : {_rel(MANIFEST)}")
    log.info("=" * 62)
    log.info("")
    log.info("Next steps:")
    log.info("  1. uv run model_a/append_real_features.py")
    log.info("  2. Delete model_a/.sm1_checkpoint.json")
    log.info("  3. uv run model_a/train.py  (only Submodel 1 retrains)")


if __name__ == "__main__":
    main()
