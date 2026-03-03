## Step 1 — Frame Extraction for Model A (Synthetic Media Detection)

## HOW IT WORKS
## ─────────────
## Instead of feeding whole videos into our model, we break each video into
## individual frames and treat each frame as an image. This is standard practice
## because deep learning models like EfficientNet-B4 are trained on images, not
## video files.

## We sample at 1 frame per second (1 fps) and cap at 30 frames per video.
## This gives us up to 30 snapshot images from the first 30 seconds of each
## video — enough to capture the visual patterns we need without blowing up
## disk space. Each frame is resized to 224×224 pixels (the standard input size
## for EfficientNet-B4) and saved as a JPEG.

## WHY SPLIT AT VIDEO LEVEL
## ─────────────────────────
## We assign each WHOLE VIDEO to either train or test before extracting any
## frames. This is critical — if we split at frame level, frames from the same
## video would appear in both train and test, the model would just memorise the
## video and report inflated accuracy. Splitting at video level ensures the
## test set contains videos the model has genuinely never seen.

## OUTPUT STRUCTURE
## ─────────────────
##   data/model_a_datasets/frames/
##       train/
##           real/
##               celeb_real/        {video_id}/  frame_0001.jpg …
##               youtube_real/      {video_id}/  …
##               dfdc_part0/        {video_id}/  …
##               dfdc_part1/        {video_id}/  …
##               dfdc_part2/        {video_id}/  …
##               faceforensics_real/{video_id}/  …
##               pexels/            {video_id}/  …
##           deepfake/
##               celeb_synthesis/        {video_id}/  …
##               dfdc_part0/             {video_id}/  …
##               dfdc_part1/             {video_id}/  …
##               dfdc_part2/             {video_id}/  …
##               faceforensics_deepfake/ {video_id}/  …
##           ai_generated/
##               videocraft/       {video_id}/  …
##               animatediff/      {video_id}/  …
##               cogvideox/        {video_id}/  …
##               runwayml/         {video_id}/  …
##               stable_diffusion/ {video_id}/  …
##               videopoet/        {video_id}/  …
##       test/
##           (same structure as train/)
##       manifest.csv  — one row per video, written as we go (crash-safe)

## If the script is interrupted, just run it again — it picks up from where
## it left off via a checkpoint file. The checkpoint is deleted when done.

## Run with:  uv run python preprocessing/extract_frames.py

import csv
import json
import logging
import random
import re
import time
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


## project paths
ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "model_a_datasets"
DEEPFAKE = DATA / "deepfake_datasets"
AI_DATA  = DATA / "ai_datasets"
OUT      = DATA / "frames"
MANIFEST = OUT / "manifest.csv"
CKPT     = OUT / ".extract_checkpoint.json"


## settings
SEED         = 42
TEST_RATIO   = 0.20   ## 20% of videos go to test, 80% to train
MAX_FRAMES   = 30     ## max frames per video (1 fps, so = 30 seconds)
IMG_SIZE     = 224    ## resize every frame to 224×224
JPEG_QUALITY = 95

## batch processing — process this many videos, then pause before continuing.
## keeps the laptop from being pinned at 100% CPU for hours on end.
BATCH_SIZE   = 50     ## videos per batch
BATCH_PAUSE  = 8      ## seconds to rest between batches


## columns written to manifest.csv — one row per video
MANIFEST_FIELDS = [
    "video_id",
    "dataset_source",
    "label",
    "split",
    "frame_count",
    "frame_dir",             ## relative to project root
    "original_video_path",   ## relative to project root
]


def _rel(path: Path) -> str:
    ## return path relative to project root so manifest.csv is portable
    return str(path.relative_to(ROOT))


def _sanitize(name: str) -> str:
    ## some videocraft filenames have spaces or special chars — clean them up
    ## so they're safe to use as folder names on any OS
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")


## each collect_* function scans one dataset folder and returns a list of
## dicts, one per video, with keys: video_path, label, dataset_source, video_id

def collect_celeb_df_v2() -> list[dict]:
    base    = DEEPFAKE / "celeb_df_v2"
    entries = []

    real_dir = base / "Celeb-real"
    if real_dir.exists():
        for mp4 in sorted(real_dir.glob("*.mp4")):
            entries.append({
                "video_path":     mp4,
                "label":          "real",
                "dataset_source": "celeb_real",
                "video_id":       mp4.stem,
            })
    else:
        log.warning("  celeb_df_v2/Celeb-real not found — skipping")

    yt_dir = base / "YouTube-real"
    if yt_dir.exists():
        for mp4 in sorted(yt_dir.glob("*.mp4")):
            entries.append({
                "video_path":     mp4,
                "label":          "real",
                "dataset_source": "youtube_real",
                "video_id":       mp4.stem,
            })
    else:
        log.warning("  celeb_df_v2/YouTube-real not found — skipping")

    synth_dir = base / "Celeb-synthesis"
    if synth_dir.exists():
        for mp4 in sorted(synth_dir.glob("*.mp4")):
            entries.append({
                "video_path":     mp4,
                "label":          "deepfake",
                "dataset_source": "celeb_synthesis",
                "video_id":       mp4.stem,
            })
    else:
        log.warning("  celeb_df_v2/Celeb-synthesis not found — skipping")

    log.info(f"  celeb_df_v2          : {len(entries):>5} videos")
    return entries


def collect_dfdc() -> list[dict]:
    ## DFDC doesn't encode labels in filenames — they're in metadata.json
    ## each part has its own metadata.json with {filename: {label: FAKE/REAL}}
    _PART_MAP = {
        "dfdc_train_part_0": "dfdc_part0",
        "dfdc_train_part_1": "dfdc_part1",
        "dfdc_train_part_2": "dfdc_part2",
    }
    entries = []

    for part_folder, source_name in _PART_MAP.items():
        part_dir  = DEEPFAKE / "dfdc" / part_folder
        meta_file = part_dir / "metadata.json"

        if not part_dir.exists():
            log.warning(f"  dfdc/{part_folder} not found — skipping")
            continue
        if not meta_file.exists():
            log.warning(f"  dfdc/{part_folder}/metadata.json missing — skipping")
            continue

        meta       = json.loads(meta_file.read_text())
        part_count = 0
        for filename, info in meta.items():
            mp4 = part_dir / filename
            if not mp4.exists():
                continue
            label = "deepfake" if info.get("label", "").upper() == "FAKE" else "real"
            entries.append({
                "video_path":     mp4,
                "label":          label,
                "dataset_source": source_name,
                "video_id":       mp4.stem,
            })
            part_count += 1
        log.info(f"  {source_name:<22} : {part_count:>5} videos")

    log.info(f"  dfdc total           : {len(entries):>5} videos")
    return entries


def collect_faceforensics() -> list[dict]:
    ## FaceForensics++ has two clear folders: original (real) and
    ## manipulated/Deepfakes (fake). Labels come from folder structure alone.
    base    = DEEPFAKE / "faceforensics"
    entries = []

    real_dir = base / "original_sequences" / "youtube" / "c23" / "videos"
    if real_dir.exists():
        for mp4 in sorted(real_dir.glob("*.mp4")):
            entries.append({
                "video_path":     mp4,
                "label":          "real",
                "dataset_source": "faceforensics_real",
                "video_id":       mp4.stem,
            })
    else:
        log.warning("  faceforensics original_sequences not found — skipping")

    fake_dir = base / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    if fake_dir.exists():
        for mp4 in sorted(fake_dir.glob("*.mp4")):
            entries.append({
                "video_path":     mp4,
                "label":          "deepfake",
                "dataset_source": "faceforensics_deepfake",
                "video_id":       mp4.stem,
            })
    else:
        log.warning("  faceforensics manipulated_sequences not found — skipping")

    log.info(f"  faceforensics        : {len(entries):>5} videos")
    return entries


## maps GenVideo generator folder names to clean short source names
_GENERATOR_MAP = {
    "BDAnimateDiffLightning": "animatediff",
    "CogVideoX5B":            "cogvideox",
    "RunwayML":               "runwayml",
    "StableDiffusion":        "stable_diffusion",
    "VideoPoet":              "videopoet",
}


def collect_genvideo() -> list[dict]:
    base    = AI_DATA / "GenVideo"
    entries = []

    ## videocraft: flat folder of AI-generated mp4s
    vc_dir = base / "AIGVDet" / "T2V" / "videocraft_mp4"
    if vc_dir.exists():
        vc_count = 0
        for mp4 in sorted(vc_dir.glob("*.mp4")):
            entries.append({
                "video_path":     mp4,
                "label":          "ai_generated",
                "dataset_source": "videocraft",
                "video_id":       _sanitize(mp4.stem),
            })
            vc_count += 1
        log.info(f"  videocraft           : {vc_count:>5} videos")
    else:
        log.warning("  GenVideo/AIGVDet/T2V/videocraft_mp4 not found — skipping")

    deepaction = base / "deepaction"

    ## Pexels: real stock videos. Each action folder has one file: a.mp4
    ## video_id = the action folder name (e.g. 000, 001, ...)
    pexels_dir = deepaction / "Pexels"
    if pexels_dir.exists():
        pexels_count = 0
        for action_dir in sorted(pexels_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            mp4 = action_dir / "a.mp4"
            if mp4.exists():
                entries.append({
                    "video_path":     mp4,
                    "label":          "real",
                    "dataset_source": "pexels",
                    "video_id":       action_dir.name,
                })
                pexels_count += 1
        log.info(f"  pexels               : {pexels_count:>5} videos")
    else:
        log.warning("  GenVideo/deepaction/Pexels not found — skipping")

    ## AI generators: each has action folders, each action has 5 variants (b–e)
    ## video_id = {action_id}_{variant}  e.g. 000_b, 001_c
    for gen_folder, source_name in _GENERATOR_MAP.items():
        gen_dir = deepaction / gen_folder
        if not gen_dir.exists():
            log.warning(f"  deepaction/{gen_folder} not found — skipping")
            continue
        gen_count = 0
        for action_dir in sorted(gen_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            for mp4 in sorted(action_dir.glob("*.mp4")):
                entries.append({
                    "video_path":     mp4,
                    "label":          "ai_generated",
                    "dataset_source": source_name,
                    "video_id":       f"{action_dir.name}_{mp4.stem}",
                })
                gen_count += 1
        log.info(f"  {source_name:<22} : {gen_count:>5} videos")

    return entries


def assign_splits(entries: list[dict]) -> list[dict]:
    ## shuffle and split within each label class so that train and test both
    ## have the same proportion of real / deepfake / ai_generated videos
    by_label: dict[str, list] = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)

    rng = random.Random(SEED)
    log.info("Split (80% train / 20% test, stratified per label):")
    for label in sorted(by_label):
        group  = by_label[label]
        rng.shuffle(group)
        n_test = max(1, int(len(group) * TEST_RATIO))
        for i, e in enumerate(group):
            e["split"] = "test" if i < n_test else "train"
        log.info(f"  {label:<14} {len(group) - n_test:>5} train  /  {n_test:>4} test")

    return entries


def extract_frames(entry: dict) -> int:
    ## open the video, figure out its FPS, then seek to the frame at second 0,
    ## second 1, second 2 ... up to MAX_FRAMES. Resize each to 224×224 and
    ## save as JPEG. Returns how many frames were actually saved.
    out_dir = (
        OUT
        / entry["split"]
        / entry["label"]
        / entry["dataset_source"]
        / entry["video_id"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(entry["video_path"]))
    if not cap.isOpened():
        log.warning(f"    cannot open: {entry['video_path']}")
        return 0

    try:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / fps

        n_seconds      = min(int(duration_s), MAX_FRAMES)
        sample_indices = [
            int(s * fps)
            for s in range(n_seconds)
            if int(s * fps) < total_frames
        ]

        if not sample_indices:
            log.warning(f"    video too short or unreadable: {entry['video_path']}")
            return 0

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


def load_checkpoint() -> set[str]:
    if CKPT.exists():
        return set(json.loads(CKPT.read_text()).get("done", []))
    return set()


def save_checkpoint(done: set[str]):
    CKPT.write_text(json.dumps({"done": list(done)}, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Discovering videos across all datasets ...")
    entries = []
    entries += collect_celeb_df_v2()
    entries += collect_dfdc()
    entries += collect_faceforensics()
    entries += collect_genvideo()
    log.info(f"Total videos found     : {len(entries):>5}")
    log.info("")

    entries = assign_splits(entries)
    log.info("")

    done = load_checkpoint()
    all_uids = {f"{e['dataset_source']}__{e['video_id']}" for e in entries}
    if all_uids and all_uids.issubset(done):
        log.info("All videos already processed — nothing to do. Exiting.")
        return
    if done:
        log.info(f"Resuming — {len(done)} videos already done, skipping them.")

    manifest_exists = MANIFEST.exists()
    manifest_file   = open(MANIFEST, "a", newline="", encoding="utf-8")
    writer          = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    if not manifest_exists:
        writer.writeheader()

    total              = len(entries)
    n_done             = len(done)
    n_skipped_at_start = len(done)   ## how many were already done before this run
    n_success          = 0
    n_failed           = 0
    t_start            = time.time()

    log.info("=" * 70)
    log.info("Extracting frames ...")
    log.info("")

    try:
        for entry in entries:
            uid = f"{entry['dataset_source']}__{entry['video_id']}"
            if uid in done:
                continue

            n_done += 1
            t0      = time.time()

            try:
                frame_count = extract_frames(entry)
            except Exception as exc:
                log.error(f"  [{n_done}/{total}]  ERROR — {uid}: {exc}")
                frame_count = 0

            if frame_count > 0:
                writer.writerow({
                    "video_id":            entry["video_id"],
                    "dataset_source":      entry["dataset_source"],
                    "label":               entry["label"],
                    "split":               entry["split"],
                    "frame_count":         frame_count,
                    "frame_dir":           _rel(
                        OUT / entry["split"] / entry["label"]
                            / entry["dataset_source"] / entry["video_id"]
                    ),
                    "original_video_path": _rel(entry["video_path"]),
                })
                manifest_file.flush()
                n_success += 1

                elapsed = time.time() - t0
                log.info(
                    f"  [{n_done:>5}/{total}]  "
                    f"{entry['split']:<5}  "
                    f"{entry['label']:<14}  "
                    f"{entry['dataset_source']:<24}  "
                    f"{entry['video_id']:<35}  "
                    f"{frame_count:>2} frames  [{elapsed:.1f}s]"
                )
            else:
                n_failed += 1
                log.warning(
                    f"  [{n_done:>5}/{total}]  SKIP (0 frames)  "
                    f"{entry['dataset_source']} / {entry['video_id']}"
                )

            done.add(uid)
            save_checkpoint(done)

            ## after every BATCH_SIZE videos, take a short rest so the laptop
            ## has a chance to cool down before the next batch begins automatically.
            ## the script continues on its own — you don't need to restart anything.
            if (n_done - n_skipped_at_start) % BATCH_SIZE == 0:
                elapsed_total = time.time() - t_start
                rate          = elapsed_total / max(n_done - n_skipped_at_start, 1)
                eta_s         = (total - n_done) * rate
                remaining     = total - n_done
                log.info("")
                log.info(
                    f"  Batch complete. {n_done}/{total} done  |  "
                    f"remaining {remaining}  |  "
                    f"elapsed {elapsed_total / 60:.1f} min  |  "
                    f"ETA ~{eta_s / 60:.0f} min"
                )
                if remaining > 0:
                    log.info(f"  Pausing {BATCH_PAUSE}s before next batch ...")
                    time.sleep(BATCH_PAUSE)
                    log.info("  Resuming ...")
                log.info("")

    finally:
        manifest_file.close()

    if len(done) >= total:
        CKPT.unlink(missing_ok=True)
        log.info("Checkpoint removed — all videos processed.")

    log.info("")
    log.info("=" * 70)
    log.info("Frame extraction complete.")
    log.info(f"  total videos    : {total}")
    log.info(f"  frames saved    : {n_success} videos with ≥ 1 frame")
    log.info(f"  skipped (empty) : {n_failed}")
    log.info(f"  manifest        : {_rel(MANIFEST)}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
