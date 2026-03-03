## Organise original video files into train / test folders.
##
## Reads manifest.csv (which already records the train/test split per video),
## then physically moves each video file into:
##
##   data/model_a_datasets/videos/
##       train/
##           real/celeb_real/id0_0001.mp4
##           deepfake/dfdc_part0/abc.mp4
##           ai_generated/cogvideox/xyz.mp4
##           ...
##       test/
##           (same structure)
##
## After moving, manifest.csv is updated so original_video_path points to
## the new location. Safe to re-run — already-moved files are skipped.
##
## Run with:  uv run python preprocessing/split_videos.py

import csv
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / "data" / "model_a_datasets"
MANIFEST = DATA / "frames" / "manifest.csv"
OUT      = DATA / "videos"


def main():
    log.info("=" * 60)
    log.info("Video train/test split — moving files")
    log.info("=" * 60)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    n_moved   = 0
    n_skipped = 0   ## already in the right place
    n_missing = 0   ## source file not found

    for row in rows:
        src = ROOT / row["original_video_path"]

        ## figure out where this video should live after the move
        dst_dir = OUT / row["split"] / row["label"] / row["dataset_source"]
        dst     = dst_dir / src.name

        ## already moved in a previous run — just make sure manifest is current
        if not src.exists() and dst.exists():
            row["original_video_path"] = str(dst.relative_to(ROOT))
            n_skipped += 1
            continue

        if not src.exists():
            log.warning(f"  missing: {row['original_video_path']} — skipping")
            n_missing += 1
            continue

        ## already at the destination (same file, same path) — nothing to do
        if src == dst:
            n_skipped += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        row["original_video_path"] = str(dst.relative_to(ROOT))
        n_moved += 1

        if n_moved % 500 == 0:
            log.info(f"  moved {n_moved} so far ...")

    ## write updated manifest back
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("")
    log.info(f"  moved   : {n_moved}")
    log.info(f"  skipped : {n_skipped}  (already in place)")
    log.info(f"  missing : {n_missing}  (source not found)")
    log.info(f"  manifest updated: {MANIFEST.relative_to(ROOT)}")
    log.info("=" * 60)
    log.info("Done.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
