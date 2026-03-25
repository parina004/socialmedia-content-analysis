## agent/tools.py
## The 6 tools the LangChain ReAct agent can call.
## Each @tool decorated function is one action the agent can choose to take.
## The docstring of each tool is what the agent reads to decide when to use it.

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from langchain.tools import tool

from model_a import inference as model_a_inference
from model_b import inference as model_b_inference
from rag.retriever import retrieve
from llm.llm_client import generate
from llm.prompts import format_synthetic_prompt, format_virality_prompt


## ── Tool 1 ────────────────────────────────────────────────────────────────

@tool
def run_synthetic_detection(video_path: str) -> str:
    """
    Runs deepfake and AI-generation detection on a video file.
    Always call this first when a video needs authenticity analysis.
    Input: absolute path to a video file.
    Returns: label (Real/Deepfake/AI-Generated), confidence score, and probabilities.
    """
    result = model_a_inference.predict(Path(video_path))
    return (
        f"Label: {result['label']} | "
        f"Confidence: {result['confidence']} | "
        f"AI-prob: {result['prob_ai']} | "
        f"Deepfake-prob: {result['prob_deepfake']}"
    )


## ── Tool 2 ────────────────────────────────────────────────────────────────

@tool
def run_virality_prediction(input_json: str) -> str:
    """
    Runs virality prediction on a video and returns a score from 0 to 100.
    Input: a JSON string with keys: video_path, title, post_hour (0-23),
    post_day (0-6 where 0=Monday), tag_count.
    Returns: JSON string with virality score, label, probability, and all features.
    """
    data   = json.loads(input_json)
    result = model_b_inference.predict(
        video_path = Path(data["video_path"]),
        title      = data["title"],
        post_hour  = int(data["post_hour"]),
        post_day   = int(data["post_day"]),
        tag_count  = int(data["tag_count"]),
    )
    ## Return as JSON so generate_virality_report can parse all individual feature values
    return json.dumps({
        "virality_score": result["virality_score"],
        "label":          result["label"],
        "probability":    result["probability"],
        "top_features":   result["top_features"],
        "features":       result["features"],
    })


## ── Tool 3 ────────────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the research knowledge base for relevant context.
    Use this before writing any report, or when you need to ground your
    reasoning in research papers and technical documentation.
    Input: a natural language search query.
    Returns: the top 5 most relevant chunks from the knowledge base.
    """
    return retrieve(query, k=5)


## ── Tool 4 ────────────────────────────────────────────────────────────────

@tool
def fetch_trending_hashtags(topic: str) -> str:
    """
    Fetches currently trending hashtags for a given topic.
    Use this to provide relevant hashtag suggestions in virality reports.
    Input: a topic string e.g. 'fitness', 'cooking', 'gaming', 'news'.
    Returns: a list of trending hashtags for that topic.
    """
    import os
    from googleapiclient.discovery import build

    api_key = os.getenv("YOUTUBE_API_KEY")

    ## Fallback if no API key is set
    if not api_key:
        fallback = {
            "fitness":  ["#fitness", "#workout", "#gym", "#health", "#motivation"],
            "cooking":  ["#cooking", "#food", "#recipe", "#foodie", "#homecook"],
            "gaming":   ["#gaming", "#gamer", "#gameplay", "#streamer", "#twitch"],
            "news":     ["#news", "#breakingnews", "#trending", "#viral", "#today"],
        }
        tags = fallback.get(topic.lower(), ["#viral", "#trending", "#fyp", "#foryou"])
        return f"Trending hashtags for '{topic}': {' '.join(tags)}"

    youtube  = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part       = "snippet",
        chart      = "mostPopular",
        maxResults = 10,
        regionCode = "US",
    ).execute()

    tags = []
    for item in response.get("items", []):
        tags.extend(item["snippet"].get("tags", []))

    seen, hashtags = set(), []
    for t in tags:
        clean = "#" + t.lower().replace(" ", "")
        if clean not in seen:
            seen.add(clean)
            hashtags.append(clean)
        if len(hashtags) >= 15:
            break

    if not hashtags:
        hashtags = ["#viral", "#trending", "#fyp", "#foryou", "#explore"]

    return f"Trending hashtags for '{topic}': {' '.join(hashtags)}"


## ── Tool 5 ────────────────────────────────────────────────────────────────

@tool
def generate_forensic_report(detection_result: str) -> str:
    """
    Writes a forensic report explaining synthetic media detection results.
    Call this after run_synthetic_detection.
    Internally queries the knowledge base and uses the LLM to write the report.
    Input: the full output string from run_synthetic_detection.
    Returns: a written forensic report in clear prose.
    """
    parts      = dict(item.split(": ") for item in detection_result.split(" | "))
    label      = parts.get("Label", "Unknown")
    confidence = float(parts.get("Confidence", 0.5))
    prob_ai    = float(parts.get("AI-prob", 0.0))
    prob_df    = float(parts.get("Deepfake-prob", 0.0))

    ## Adapt the RAG query based on what the model actually found
    if label == "AI-Generated":
        query = "AI video generation artifacts diffusion model detection"
    elif label == "Deepfake":
        query = "deepfake facial manipulation detection forensic evidence"
    elif confidence < 0.65:
        query = "borderline inconclusive synthetic media detection threshold"
    else:
        query = "real authentic video detection synthetic media"

    rag_context = retrieve(query, k=5)

    prompt = format_synthetic_prompt(
        label              = label,
        confidence         = confidence * 100,   ## prompt expects percentage e.g. 97.85
        efficientnet_score = prob_ai,
        forensic_score     = prob_df,
        face_score         = None,
        rag_context        = rag_context,
    )

    return generate(prompt)


## ── Tool 6 ────────────────────────────────────────────────────────────────

@tool
def generate_virality_report(input_json: str) -> str:
    """
    Writes a virality analysis report with actionable improvement tips.
    Call this after run_virality_prediction.
    Input: a JSON string with keys: virality_result (the full JSON string
    from run_virality_prediction), user_caption, user_hashtags.
    Returns: a written virality analysis report with specific improvement tips.
    """
    data            = json.loads(input_json)
    virality_result = json.loads(data["virality_result"])
    user_caption    = data.get("user_caption", "")
    user_hashtags   = data.get("user_hashtags", "")

    f = virality_result["features"]

    day_names  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    upload_day = day_names[int(f.get("upload_day", 0))] if f.get("upload_day", 0) >= 0 else "Unknown"

    rag_context = retrieve("virality prediction social media engagement features", k=5)

    prompt = format_virality_prompt(
        virality_label        = virality_result["label"],
        viral_probability     = virality_result["probability"] * 100,
        engagement_percentile = float(virality_result["virality_score"]),
        brisque               = float(f.get("brisque_score", 0.0)),
        vibrancy              = float(f.get("color_vibrancy", 0.0)),
        motion                = float(f.get("motion_intensity", 0.0)),
        face_ratio            = float(f.get("face_presence_ratio", 0.0)),
        tempo                 = float(f.get("tempo_bpm", 0.0)),
        rms_energy            = float(f.get("rms_energy", 0.0)),
        speech_ratio          = float(f.get("speech_ratio", 0.0)),
        title_sentiment       = float(f.get("title_sentiment", 0.0)),
        title_length          = int(f.get("title_length", 0)),
        tag_count             = int(f.get("tag_count", 0)),
        upload_hour           = int(f.get("upload_hour", 0)),
        upload_day            = upload_day,
        user_caption          = user_caption,
        user_hashtags         = user_hashtags,
        rag_context           = rag_context,
    )

    return generate(prompt)
