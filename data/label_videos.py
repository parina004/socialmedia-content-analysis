## We have raw videos in videos.csv. This script decides which ones are
## "viral" and which ones are "not_viral" — and throws away the ones in
## the middle that are too ambiguous to be useful for training.
##
## Labelling is done within each category separately. This matters because
## a cooking video with 500K views might be viral for cooking, but a music
## video with 500K views is just average. We compare apples to apples.
##
## Labelling rule per category (sorted by view count):
##   top 20%    -> viral
##   bottom 40% -> not_viral
##   middle 40% -> deleted (too ambiguous, would confuse the model)
##
## We also compute two ratio features here that the model needs:
##   like_to_view_ratio    = likes / views   (how much people liked it)
##   comment_to_view_ratio = comments / views (how much people talked about it)
##
## Ratios matter more than raw counts — a video with 10K likes and 10M views
## is less engaging than one with 5K likes and 50K views. The ratio tells the truth.
##
## Run with:  uv run python data/label_videos.py

import csv
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


## file paths

BASE   = Path(__file__).parent
INPUT  = BASE / "videos.csv"
OUTPUT = BASE / "labelled_videos.csv"


## labelling thresholds

## sorted ascending by view count within each category:
##   below NOTVIRAL_CUTOFF = not_viral
##   above VIRAL_CUTOFF    = viral
##   everything in between = deleted
VIRAL_CUTOFF    = 0.80  ## top 20%
NOTVIRAL_CUTOFF = 0.40  ## bottom 40%


## output columns — same as videos.csv plus the label and two ratio features

OUTPUT_FIELDS = [
    "video_id", "title", "description", "tags",
    "view_count", "like_count", "comment_count",
    "duration_seconds", "category", "upload_date",
    "like_to_view_ratio",     ## computed here
    "comment_to_view_ratio",  ## computed here
    "label",                  ## viral or not_viral
]


## main

def label():
    if not INPUT.exists():
        raise FileNotFoundError("videos.csv not found — run collect_youtube.py first.")

    ## load all videos into memory, grouped by category
    log.info("Loading videos.csv ...")
    cat_groups = defaultdict(list)

    with open(INPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ## convert numeric fields from string to int
            row["view_count"]    = int(row["view_count"])
            row["like_count"]    = int(row["like_count"])
            row["comment_count"] = int(row["comment_count"])

            ## compute the two ratio features while we have the raw counts
            ## if a video has 0 views we set ratios to 0 to avoid division by zero
            views = row["view_count"]
            row["like_to_view_ratio"]    = round(row["like_count"]    / views, 6) if views > 0 else 0.0
            row["comment_to_view_ratio"] = round(row["comment_count"] / views, 6) if views > 0 else 0.0

            cat_groups[row["category"]].append(row)

    log.info(f"Loaded {sum(len(v) for v in cat_groups.values())} videos across {len(cat_groups)} categories")

    ## label each category separately
    labelled   = []  ## rows we keep (viral + not_viral)
    n_viral    = 0
    n_notviral = 0
    n_deleted  = 0

    log.info("Labelling within each category ...")

    for cat, videos in sorted(cat_groups.items()):
        ## sort ascending by view count so index 0 = least viewed
        videos.sort(key=lambda r: r["view_count"])
        n = len(videos)

        notviral_end = int(n * NOTVIRAL_CUTOFF)  ## everything below this index = not_viral
        viral_start  = int(n * VIRAL_CUTOFF)      ## everything at or above this index = viral

        cat_viral    = 0
        cat_notviral = 0
        cat_deleted  = 0

        for i, row in enumerate(videos):
            if i < notviral_end:
                row["label"] = "not_viral"
                labelled.append(row)
                cat_notviral += 1
            elif i >= viral_start:
                row["label"] = "viral"
                labelled.append(row)
                cat_viral += 1
            else:
                ## middle 40% — drop this row entirely
                cat_deleted += 1

        n_viral    += cat_viral
        n_notviral += cat_notviral
        n_deleted  += cat_deleted

        log.info(
            f"  [{cat:<12}]  total={n:>4}  "
            f"viral={cat_viral:>3}  not_viral={cat_notviral:>3}  deleted={cat_deleted:>3}"
        )

    ## write the labelled rows to the output CSV
    log.info(f"Writing {len(labelled)} labelled videos to labelled_videos.csv ...")

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in labelled:
            writer.writerow({field: row[field] for field in OUTPUT_FIELDS})

    ## print a summary
    total_kept = n_viral + n_notviral
    log.info("")
    log.info("Labelling complete.")
    log.info(f"  viral     : {n_viral:>5}  ({n_viral / total_kept * 100:.1f}%)")
    log.info(f"  not_viral : {n_notviral:>5}  ({n_notviral / total_kept * 100:.1f}%)")
    log.info(f"  deleted   : {n_deleted:>5}  (middle 40% per category — too ambiguous)")
    log.info(f"  total kept: {total_kept:>5}")
    log.info(f"  saved to  : {OUTPUT}")


if __name__ == "__main__":
    label()
