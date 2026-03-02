## collect_youtube.py already ran and got most categories to 500 videos,
## but some categories fell short because the original search queries were
## too generic and YouTube ran out of pages. This script tops them up
## using multiple specific search queries so we don't repeat the same
## query that already ran dry.

## It also adds 4 brand new categories from scratch: finance, fitness,
## true_crime, and travel.

## It reads videos.csv first to know which video IDs we already have,
## then appends new unique ones until each category reaches 500.

## Run with:  uv run python data/topup_youtube.py

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

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


## file paths — same CSV as the main collection script

BASE   = Path(__file__).parent
OUTPUT = BASE / "videos.csv"


## api keys — same rotation logic as collect_youtube.py

API_KEYS = [
    k for k in [
        os.getenv("YT_API_1"),
        os.getenv("YT_API_2"),
        os.getenv("YT_API_3"),
    ] if k
]


## settings

TARGET_PER_CAT    = 500   ## we want 500 per category just like the other categories
PAGE_SIZE         = 50
SLEEP_BETWEEN_REQ = 0.5


## each category gets multiple search queries — we cycle through them so if
## one query runs out of pages we move to the next one instead of giving up.
## categories already at 500 are auto-skipped when the script runs.
TOPUP_CATEGORIES = [

    ## existing categories that are still short

    {
        "name": "news",
        "queries": [
            "news breaking report today",
            "world news update 2024",
            "political news analysis",
            "investigative journalism report",
            "local news broadcast",
            "international news headlines",
            "news documentary",
            "election news coverage",
            "war conflict news report",
            "economic news update",
        ],
    },
    {
        "name": "cooking",
        "queries": [
            "cooking recipe easy dinner",
            "baking tutorial bread",
            "chef cooking techniques",
            "street food recipe",
            "healthy meal prep",
            "dessert recipe chocolate",
            "asian food recipe",
            "italian pasta recipe",
            "quick 15 minute recipe",
            "cooking challenge food",
        ],
    },
    {
        "name": "vlogs",
        "queries": [
            "day in my life vlog",
            "weekly vlog lifestyle",
            "college student vlog",
            "moving vlog new city",
            "morning routine vlog",
            "grwm get ready with me",
            "productive day vlog",
            "vlog challenge week",
            "couple vlog daily life",
            "solo vlog everyday",
        ],
    },
    {
        "name": "music",
        "queries": [
            "official music video 2024",
            "new song release music",
            "acoustic cover song",
            "hip hop rap music video",
            "pop music official video",
            "r&b soul music video",
            "indie music official",
            "live performance concert",
            "album release music video",
            "remix official music video",
        ],
    },
    {
        "name": "technology",
        "queries": [
            "tech review smartphone 2024",
            "unboxing new gadget",
            "AI technology explained",
            "laptop review comparison",
            "technology news update",
            "coding tutorial programming",
            "gadget review tech",
            "software review tutorial",
            "tech tips tricks",
            "future technology explained",
        ],
    },
    {
        "name": "education",
        "queries": [
            "educational explainer science",
            "history documentary lesson",
            "learn english tutorial",
            "math explained tutorial",
            "science experiment education",
            "psychology explained",
            "economics explained beginner",
            "philosophy explained",
            "space astronomy explained",
            "biology lesson explained",
        ],
    },

    ## brand new categories being added from scratch

    {
        "name": "finance",
        "queries": [
            "investing stocks beginners",
            "personal finance money tips",
            "how to save money",
            "stock market explained",
            "passive income ideas",
            "cryptocurrency bitcoin explained",
            "real estate investing",
            "budgeting money management",
            "financial independence retire early",
            "how to build wealth",
        ],
    },
    {
        "name": "fitness",
        "queries": [
            "workout routine gym beginner",
            "home workout no equipment",
            "weight loss exercise",
            "muscle building training",
            "yoga tutorial beginner",
            "cardio workout routine",
            "HIIT workout intense",
            "strength training program",
            "fitness transformation journey",
            "healthy lifestyle workout",
        ],
    },
    {
        "name": "true_crime",
        "queries": [
            "true crime documentary case",
            "unsolved mystery crime",
            "murder mystery documentary",
            "criminal case explained",
            "detective investigation true crime",
            "cold case solved documentary",
            "serial killer documentary",
            "true crime story retold",
            "prison documentary crime",
            "crime investigation explained",
        ],
    },
    {
        "name": "travel",
        "queries": [
            "travel vlog destination",
            "solo travel adventure",
            "budget travel tips",
            "travel guide city tour",
            "backpacking travel adventure",
            "luxury travel experience",
            "travel documentary country",
            "road trip travel vlog",
            "international travel tips",
            "hidden gems travel destination",
        ],
    },

    ## categories that are already at 500+ — kept here so the script skips them cleanly

    {
        "name": "sports",
        "queries": [
            "NFL highlights 2024",
            "cricket match highlights",
            "NBA basketball highlights",
            "soccer football goal",
            "tennis match highlights",
            "baseball game highlights",
            "sports compilation viral",
            "rugby match highlights",
            "hockey game highlights",
            "athletics sprinting championship",
        ],
    },
    {
        "name": "comedy",
        "queries": [
            "stand up comedy performance",
            "funny sketch comedy show",
            "comedian viral clip",
            "comedy skit funny",
            "fails funny compilation",
            "prank video funny",
            "improv comedy funny",
            "late night comedy clip",
            "sitcom funny moments",
            "roast comedy funny",
        ],
    },
    {
        "name": "beauty",
        "queries": [
            "makeup tutorial beginner",
            "skincare routine morning",
            "hair tutorial styling",
            "nail art tutorial",
            "foundation routine full coverage",
            "eye makeup tutorial",
            "drugstore makeup review",
            "glow up transformation",
            "contouring tutorial",
            "natural makeup look tutorial",
        ],
    },
]

CSV_FIELDS = [
    "video_id", "title", "description", "tags",
    "view_count", "like_count", "comment_count",
    "duration_seconds", "category", "upload_date",
]


## helper functions — same as collect_youtube.py

def parse_duration(iso: str) -> int:
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not match:
        return 0
    h, m, s = (int(x or 0) for x in match.groups())
    return h * 3600 + m * 60 + s


def search_page(yt, query: str, page_token: str | None) -> tuple[list[str], str | None]:
    params = {
        "q":               query,
        "type":            "video",
        "maxResults":      PAGE_SIZE,
        "order":           "relevance",
        "part":            "id",
        "videoEmbeddable": "true",
        "videoSyndicated": "true",
    }
    if page_token:
        params["pageToken"] = page_token

    resp = yt.search().list(**params).execute()
    ids  = [item["id"]["videoId"] for item in resp.get("items", [])]
    return ids, resp.get("nextPageToken")


def fetch_details(yt, video_ids: list[str], category_name: str) -> list[dict]:
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
            "tags":             "|".join(tags),
            "view_count":       int(stats.get("viewCount", 0)),
            "like_count":       int(stats.get("likeCount", 0)),
            "comment_count":    int(stats.get("commentCount", 0)),
            "duration_seconds": parse_duration(content.get("duration", "")),
            "category":         category_name,
            "upload_date":      snippet.get("publishedAt", ""),
        })
    return rows


## main top-up loop

def topup():
    if not API_KEYS:
        raise ValueError("No API keys found in .env — set YT_API_1, YT_API_2, YT_API_3.")

    ## read the existing CSV to know what we already have
    ## we track both total IDs seen (for dedup) and per-category counts (to know how many more we need)
    if not OUTPUT.exists():
        raise FileNotFoundError("videos.csv not found — run collect_youtube.py first.")

    seen      = set()  ## all video IDs already in the CSV
    cat_count = {}     ## how many videos we already have per category

    log.info("Reading existing videos.csv ...")
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add(row["video_id"])
            cat_count[row["category"]] = cat_count.get(row["category"], 0) + 1

    log.info(f"Found {len(seen)} existing videos.")
    for cat in TOPUP_CATEGORIES:
        name  = cat["name"]
        have  = cat_count.get(name, 0)
        need  = max(0, TARGET_PER_CAT - have)
        log.info(f"  {name}: have {have}, need {need} more")

    ## set up API client with key rotation
    key_index = 0

    def build_client():
        log.info(f"Using API key {key_index + 1} of {len(API_KEYS)}")
        return build("youtube", "v3", developerKey=API_KEYS[key_index])

    def rotate_key():
        nonlocal key_index
        key_index += 1
        if key_index >= len(API_KEYS):
            return False
        log.warning(f"Rotating to API key {key_index + 1} of {len(API_KEYS)}")
        return True

    yt = build_client()

    ## open CSV in append mode — we're adding to what's already there
    ## no writeheader() here — header already exists from collect_youtube.py
    csv_file = open(OUTPUT, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)

    try:
        for cat in TOPUP_CATEGORIES:
            name    = cat["name"]
            queries = cat["queries"]
            have    = cat_count.get(name, 0)
            need    = TARGET_PER_CAT - have

            if need <= 0:
                log.info(f"[{name}] already at {have} — skipping")
                continue

            log.info(f"[{name}] topping up from {have} to {TARGET_PER_CAT} ({need} needed)")
            added = 0  ## how many new videos we've added in this run for this category

            ## cycle through queries — if one runs dry, move to the next
            for query in queries:
                if added >= need:
                    break

                log.info(f"  [{name}] query: \"{query}\"")
                page_token = None

                while added < need:

                    ## search for video IDs using this query
                    try:
                        ids, page_token = search_page(yt, query, page_token)
                        time.sleep(SLEEP_BETWEEN_REQ)
                    except HttpError as e:
                        if "quotaExceeded" in str(e):
                            if rotate_key():
                                yt = build_client()
                                continue
                            log.warning("All API keys exhausted. Save current progress and stop.")
                            csv_file.close()
                            return
                        raise

                    ## filter out video IDs we already have
                    new_ids = [vid for vid in ids if vid not in seen]

                    if not new_ids:
                        ## this page had only duplicates — move to next page or next query
                        if not page_token:
                            log.info(f"  [{name}] query \"{query}\" exhausted")
                            break
                        continue

                    ## fetch the full details for this batch
                    batch = new_ids[:50]
                    try:
                        rows = fetch_details(yt, batch, name)
                        time.sleep(SLEEP_BETWEEN_REQ)
                    except HttpError as e:
                        if "quotaExceeded" in str(e):
                            if rotate_key():
                                yt = build_client()
                                continue
                            log.warning("All API keys exhausted. Save current progress and stop.")
                            csv_file.close()
                            return
                        raise

                    ## write to CSV immediately
                    for row in rows:
                        if added >= need:
                            break
                        writer.writerow(row)
                        seen.add(row["video_id"])
                        added += 1

                    log.info(f"  [{name}]  {have + added}/{TARGET_PER_CAT}")

                    if not page_token:
                        log.info(f"  [{name}] query \"{query}\" exhausted")
                        break

            ## report final result for this category
            final = have + added
            if final >= TARGET_PER_CAT:
                log.info(f"[{name}] reached {final} videos")
            else:
                log.warning(f"[{name}] only reached {final} videos — all queries exhausted")

    finally:
        csv_file.close()

    log.info(f"Top-up complete. Total videos in CSV: {len(seen)}")


if __name__ == "__main__":
    topup()
