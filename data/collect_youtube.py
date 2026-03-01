## We need 5,000 YouTube videos for training Model B (virality prediction).
## This script talks to the YouTube API, searches 10 different categories,
## grabs 500 videos per category, and saves everything to data/videos.csv.
##
## YouTube gives you 10,000 free API "units" per day per account.
## We have 3 API keys, so 30,000 units total — enough to finish in one run.
## If one key runs out mid-way, we automatically switch to the next one.
## If ALL keys run out (shouldn't happen), we save progress and stop —
## just run the script again tomorrow and it picks up where it left off.
##
## Run with:  uv run python data/collect_youtube.py

import os
import csv
import time
import json
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

## read the .env file so os.getenv() can find our API keys
load_dotenv()

## set up logging so we can see what's happening while the script runs overnight
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


## file paths

BASE   = Path(__file__).parent                        ## this is the /data folder
OUTPUT = BASE / "videos.csv"                          ## where we save all collected videos
CKPT   = BASE / ".collection_checkpoint.json"         ## saves progress in case we stop early


## api keys

## pull all 3 keys from .env — skip any that aren't set
## so if you only have 2 keys, it still works fine
API_KEYS = [
    k for k in [
        os.getenv("YT_API_1"),
        os.getenv("YT_API_2"),
        os.getenv("YT_API_3"),
    ] if k
]


## settings

VIDEOS_PER_CAT    = 500   ## how many videos we want per category
PAGE_SIZE         = 50    ## YouTube lets us fetch max 50 results per search call
SLEEP_BETWEEN_REQ = 0.5   ## wait half a second between calls so YouTube doesn't block us


## the 10 categories we search

## we use search queries instead of YouTube's category IDs because category IDs
## are too strict and often return fewer results than we need
CATEGORIES = [
    {"name": "comedy",     "query": "comedy funny video"},
    {"name": "music",      "query": "music video official"},
    {"name": "news",       "query": "news report breaking"},
    {"name": "sports",     "query": "sports highlights match"},
    {"name": "cooking",    "query": "cooking recipe tutorial"},
    {"name": "gaming",     "query": "gaming gameplay walkthrough"},
    {"name": "education",  "query": "educational tutorial learn"},
    {"name": "beauty",     "query": "beauty makeup tutorial"},
    {"name": "technology", "query": "technology review unboxing"},
    {"name": "vlogs",      "query": "daily vlog life"},
]

## these are the columns in our output CSV — one row per video
CSV_FIELDS = [
    "video_id", "title", "description", "tags",
    "view_count", "like_count", "comment_count",
    "duration_seconds", "category", "upload_date",
]


## helper functions

def parse_duration(iso: str) -> int:
    ## YouTube gives duration as "PT1H2M3S" — we convert that to plain seconds
    ## e.g. PT4M13S → 253 seconds, PT1H → 3600 seconds
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not match:
        return 0
    h, m, s = (int(x or 0) for x in match.groups())
    return h * 3600 + m * 60 + s


def load_checkpoint() -> dict:
    ## if we stopped halfway through a previous run, load where we left off
    ## if this is a fresh run, start from scratch
    if CKPT.exists():
        return json.loads(CKPT.read_text())
    return {"done": [], "seen_ids": []}


def save_checkpoint(done: list, seen_ids: list):
    ## write current progress to disk so we can resume if something goes wrong
    CKPT.write_text(json.dumps({"done": done, "seen_ids": seen_ids}, indent=2))


def search_page(yt, query: str, page_token: str | None) -> tuple[list[str], str | None]:
    ## ask YouTube to search for videos matching our query
    ## returns up to 50 video IDs and a token for the next page of results
    ##
    ## we use order=relevance (not viewCount) on purpose — if we sorted by views,
    ## every result would be a mega-viral video, and our model would only learn
    ## what extremely popular videos look like. we need a mix of popular and niche.
    ##
    ## costs 100 quota units per call — most expensive operation in this script
    params = {
        "q":               query,
        "type":            "video",
        "maxResults":      PAGE_SIZE,
        "order":           "relevance",
        "part":            "id",         ## only fetch IDs here, details come next
        "videoEmbeddable": "true",
        "videoSyndicated": "true",
    }
    if page_token:
        params["pageToken"] = page_token  ## go to next page if we're paginating

    resp = yt.search().list(**params).execute()
    ids  = [item["id"]["videoId"] for item in resp.get("items", [])]
    return ids, resp.get("nextPageToken")


def fetch_details(yt, video_ids: list[str], category_name: str) -> list[dict]:
    ## now that we have video IDs, get the actual data we need for each one:
    ## title, description, tags, view count, likes, comments, duration, upload date
    ##
    ## we can send up to 50 IDs in one call — YouTube returns all of them together
    ## this costs only 1 quota unit no matter how many IDs we send (up to 50)
    ## so we batch aggressively to save quota
    resp = yt.videos().list(
        id=",".join(video_ids),
        part="snippet,statistics,contentDetails",
    ).execute()

    rows = []
    for item in resp.get("items", []):
        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        content = item.get("contentDetails", {})
        tags    = snippet.get("tags", [])

        rows.append({
            "video_id":         item["id"],
            "title":            snippet.get("title", ""),
            "description":      snippet.get("description", ""),
            "tags":             "|".join(tags),                             ## store as pipe-separated string
            "view_count":       int(stats.get("viewCount", 0)),
            "like_count":       int(stats.get("likeCount", 0)),             ## 0 if creator disabled likes
            "comment_count":    int(stats.get("commentCount", 0)),          ## 0 if creator disabled comments
            "duration_seconds": parse_duration(content.get("duration", "")),
            "category":         category_name,
            "upload_date":      snippet.get("publishedAt", ""),
        })
    return rows


## main collection loop

def collect():
    if not API_KEYS:
        raise ValueError("No API keys found in .env — set YT_API_1, YT_API_2, YT_API_3.")

    key_index = 0  ## we start with the first key

    def build_client():
        ## create a YouTube API client using whichever key we're currently on
        log.info(f"Using API key {key_index + 1} of {len(API_KEYS)}")
        return build("youtube", "v3", developerKey=API_KEYS[key_index])

    def rotate_key():
        ## called when the current key hits its daily limit
        ## moves to the next key — returns True if there's another key, False if we're out
        nonlocal key_index
        key_index += 1
        if key_index >= len(API_KEYS):
            return False  ## all keys are done
        log.warning(f"Rotating to API key {key_index + 1} of {len(API_KEYS)}")
        return True

    yt    = build_client()
    state = load_checkpoint()
    done  = state["done"]           ## categories we already finished
    seen  = set(state["seen_ids"])  ## video IDs we already saved (avoids duplicates)

    ## open the CSV in append mode so we don't overwrite data from previous runs
    file_exists = OUTPUT.exists()
    csv_file = open(OUTPUT, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()  ## only write the header row on the very first run

    try:
        for cat in CATEGORIES:
            name = cat["name"]

            ## skip this category if we already collected it in a previous run
            if name in done:
                log.info(f"[{name}] already complete — skipping")
                continue

            log.info(f"[{name}] starting  (target {VIDEOS_PER_CAT} videos)")
            collected  = []
            page_token = None  ## no page token means start from the first page

            while len(collected) < VIDEOS_PER_CAT:

                ## step 1: search YouTube and get a page of video IDs
                try:
                    ids, page_token = search_page(yt, cat["query"], page_token)
                    time.sleep(SLEEP_BETWEEN_REQ)  ## be polite to the API
                except HttpError as e:
                    if "quotaExceeded" in str(e):
                        ## this key is out of quota for today — try the next one
                        if rotate_key():
                            yt = build_client()
                            continue  ## redo this same search with the new key
                        ## no keys left at all — save and stop
                        log.warning("All API keys exhausted. Saving checkpoint — rerun tomorrow.")
                        save_checkpoint(done, list(seen))
                        return
                    raise  ## some other API error — let it crash so we can see what's wrong

                ## filter out any video IDs we've already collected
                new_ids = [vid for vid in ids if vid not in seen]
                if not new_ids:
                    if not page_token:
                        log.warning(f"[{name}] no more pages — got {len(collected)} videos")
                        break
                    continue  ## this page had only duplicates — go to next page

                ## step 2: fetch the actual video details for this batch of IDs
                batch = new_ids[:50]
                try:
                    rows = fetch_details(yt, batch, name)
                    time.sleep(SLEEP_BETWEEN_REQ)
                except HttpError as e:
                    if "quotaExceeded" in str(e):
                        if rotate_key():
                            yt = build_client()
                            continue
                        log.warning("All API keys exhausted. Saving checkpoint — rerun tomorrow.")
                        save_checkpoint(done, list(seen))
                        return
                    raise

                ## step 3: write each video to the CSV immediately
                ## we write as we go so if the script crashes we don't lose everything
                for row in rows:
                    writer.writerow(row)
                    seen.add(row["video_id"])
                    collected.append(row["video_id"])

                log.info(f"[{name}]  {len(collected)}/{VIDEOS_PER_CAT}")

                if len(collected) >= VIDEOS_PER_CAT:
                    break

                ## YouTube ran out of pages for this search query before we hit 500
                if not page_token:
                    log.warning(f"[{name}] no more pages — got {len(collected)} videos")
                    break

            ## mark this category as done and save progress
            done.append(name)
            save_checkpoint(done, list(seen))
            log.info(f"[{name}] complete — {len(collected)} videos written")

    finally:
        csv_file.close()  ## always close the file even if something crashes

    log.info(f"All done. Total videos collected: {len(seen)}")
    log.info(f"Saved to: {OUTPUT}")
    if CKPT.exists():
        CKPT.unlink()  ## delete the checkpoint file — we don't need it anymore
        log.info("Checkpoint removed.")


if __name__ == "__main__":
    collect()
