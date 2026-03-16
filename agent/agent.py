"""
agent/agent.py
The AI Agent Orchestrator — the brain of the system.
Uses LangChain's ReAct pattern: the agent reasons about what tools to call,
calls them one by one, observes the results, and builds a final response.
Models A and B are stubbed with dummy outputs for now — real calls will be wired in Phase 5.
"""

import sys
import os
from pathlib import Path

## Add project root to path so we can import from llm/ and rag/ folders
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from groq import Groq
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from dotenv import load_dotenv

load_dotenv()

## Import our RAG retriever and LLM prompt formatters
from rag.retriever import retrieve
from llm.prompts import format_synthetic_prompt, format_virality_prompt
from llm.llm_client import generate
from model_a.predict import predict as model_a_predict
from model_b.predict import predict as model_b_predict


## Tool 1 — Run synthetic media detection on the video
## Returns a stub result for now; real Model A call will replace this in Phase 5
@tool
def run_synthetic_detection(video_path: str) -> str:
    """
    Run synthetic media detection on the given video file.
    Returns the predicted label (Real / Deepfake / AI-Generated),
    confidence score, and per-stream scores.
    Input: absolute or relative path to the video file.
    """
    ## Call the real trained Model A cascade — loads forensic_cnn.pth + submodel1/2.json
    result = model_a_predict(video_path)
    face = result["face_score"] if result["face_score"] is not None else "N/A"
    return (
        f"label: {result['label']} | confidence: {result['confidence']}% | "
        f"efficientnet_score: {result['prob_ai']} | forensic_score: {result['prob_deepfake']} | "
        f"face_score: {face}"
    )


## Tool 2 — Run virality prediction on the video
## Returns a stub result for now; real Model B call will replace this in Phase 5
@tool
def run_virality_prediction(video_path: str) -> str:
    """
    Run virality prediction on the given video file.
    Returns the predicted virality label, probability, engagement percentile,
    and all 20 extracted features.
    Input: absolute path to the video file only.
    """
    ## Strip any surrounding quotes the agent adds
    video_path = video_path.strip().strip("'\"")
    ## Call the real trained Model B LightGBM classifier with default metadata
    result = model_b_predict(
        video_path, title="", tag_count=5,
        upload_hour=12, upload_day="Monday",
    )
    f = result["features"]
    return (
        f"virality_label: {result['virality_label']} | "
        f"viral_probability: {result['viral_probability']}% | "
        f"engagement_percentile: {result['engagement_percentile']} | "
        f"brisque: {f['brisque_score']} | vibrancy: {f['color_vibrancy']} | "
        f"motion: {f['motion_intensity']} | face_ratio: {f['face_presence_ratio']} | "
        f"tempo: {f['tempo_bpm']} | rms_energy: {f['rms_energy']} | "
        f"speech_ratio: {f['speech_ratio']} | "
        f"title_sentiment: {f['title_sentiment']} | title_length: {f['title_length']} | "
        f"tag_count: {f['tag_count']} | upload_hour: {f['upload_hour']} | "
        f"upload_day: Monday"
    )


## Tool 3 — Search the RAG knowledge base for relevant research context
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the research knowledge base for information relevant to the query.
    Returns the top 5 most semantically similar text chunks from research papers
    and technical documents stored in ChromaDB.
    Use this before generating any report to add research-backed context.
    """
    ## Calls our retriever which loads ChromaDB and does cosine similarity search
    return retrieve(query, k=5)


## Tool 4 — Fetch trending hashtags for a given topic
## Uses a simple free approach — no paid API needed
@tool
def fetch_trending_hashtags(topic: str) -> str:
    """
    Fetch a list of trending hashtags relevant to the given topic.
    Returns a comma-separated string of recommended hashtags.
    Input: a content topic like 'travel', 'fitness', 'cooking', etc.
    """
    ## Hardcoded popular hashtag sets per category — good enough for demo
    ## In a real deployment this could call a hashtag analytics API
    hashtag_map = {
        "travel":   "#travel #wanderlust #explore #travelgram #adventure #vacation #trip",
        "fitness":  "#fitness #gym #workout #fitlife #health #motivation #bodybuilding",
        "food":     "#food #foodie #instafood #yummy #recipe #foodphotography #cooking",
        "tech":     "#tech #technology #AI #innovation #coding #software #gadgets",
        "fashion":  "#fashion #style #ootd #trending #outfit #aesthetic #streetwear",
        "music":    "#music #newmusic #hiphop #pop #artist #producer #beats",
        "gaming":   "#gaming #gamer #twitch #youtube #esports #videogames #ps5",
        "business": "#business #entrepreneur #startup #marketing #success #money #hustle",
    }
    ## Try to match the topic to a known category (case-insensitive)
    topic_lower = topic.lower()
    for category, tags in hashtag_map.items():
        if category in topic_lower:
            return f"Trending hashtags for '{topic}': {tags}"

    ## If no category matches, return generic viral hashtags
    return (
        f"Trending hashtags for '{topic}': "
        "#viral #trending #fyp #reels #content #explore #foryou"
    )


## Tool 5 — Generate a written forensic report using the LLM
@tool
def generate_forensic_report(model_output: str) -> str:
    """
    Generate a human-readable forensic report explaining synthetic media detection results.
    Input: the raw output string from run_synthetic_detection.
    Retrieves research context from the knowledge base, then asks the LLM to write
    a clear explanation of what the model found and what it means for the user.
    """
    ## Parse the pipe-separated key:value string from run_synthetic_detection
    ## Strip surrounding quotes the agent sometimes adds around its Action Input
    model_output = model_output.strip().strip("'\"")
    parsed = {}
    for part in model_output.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            parsed[k.strip()] = v.strip().strip("'\"").replace("%", "").strip()

    ## Pull relevant research context from the RAG knowledge base
    rag_context = retrieve("deepfake detection forensic signals EfficientNet DCT SRM", k=5)

    ## Format the full prompt and send it to the LLM
    prompt = format_synthetic_prompt(
        label=parsed.get("label", "Unknown"),
        confidence=float(parsed.get("confidence", "0")),
        efficientnet_score=float(parsed.get("efficientnet_score", "0")),
        forensic_score=float(parsed.get("forensic_score", "0")),
        face_score=float(parsed.get("face_score", "0")) if "face_score" in parsed else None,
        rag_context=rag_context,
    )
    return generate(prompt)


## Tool 6 — Generate a written virality report using the LLM
@tool
def generate_virality_report(
    model_output: str,
    user_caption: str = "",
    user_hashtags: str = "",
) -> str:
    """
    Generate a human-readable virality report with actionable content advice.
    Input: the raw output string from run_virality_prediction, plus the user's
    planned caption and hashtags (both optional).
    Retrieves research context from the knowledge base, then asks the LLM to explain
    the score and give specific suggestions for improving the video's reach.
    """
    ## Parse the pipe-separated key:value string from run_virality_prediction
    ## Strip surrounding quotes the agent sometimes adds around its Action Input
    model_output = model_output.strip().strip("'\"")
    parsed = {}
    for part in model_output.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            parsed[k.strip()] = v.strip().strip("'\"").replace("%", "").strip()

    ## Pull relevant research context from the RAG knowledge base
    rag_context = retrieve("social media virality prediction engagement features", k=5)

    ## Format the full prompt and send it to the LLM
    prompt = format_virality_prompt(
        virality_label=parsed.get("virality_label", "Unknown"),
        viral_probability=float(parsed.get("viral_probability", "0")),
        engagement_percentile=float(parsed.get("engagement_percentile", "0")),
        brisque=float(parsed.get("brisque", "0")),
        vibrancy=float(parsed.get("vibrancy", "0")),
        motion=float(parsed.get("motion", "0")),
        face_ratio=float(parsed.get("face_ratio", "0")),
        tempo=float(parsed.get("tempo", "0")),
        rms_energy=float(parsed.get("rms_energy", "0")),
        speech_ratio=float(parsed.get("speech_ratio", "0")),
        title_sentiment=float(parsed.get("title_sentiment", "0")),
        title_length=int(parsed.get("title_length", "0")),
        tag_count=int(parsed.get("tag_count", "0")),
        upload_hour=int(parsed.get("upload_hour", "12")),
        upload_day=parsed.get("upload_day", "Monday"),
        user_caption=user_caption,
        user_hashtags=user_hashtags,
        rag_context=rag_context,
    )
    return generate(prompt)


## All 6 tools collected in a list — passed to the agent below
TOOLS = [
    run_synthetic_detection,
    run_virality_prediction,
    search_knowledge_base,
    fetch_trending_hashtags,
    generate_forensic_report,
    generate_virality_report,
]


## The ReAct prompt template — tells the agent how to reason step by step
## {tools} and {tool_names} are filled in automatically by LangChain
REACT_PROMPT = PromptTemplate.from_template("""
You are an AI media analysis assistant. You help users understand whether a video is \
synthetic (deepfake or AI-generated) and how likely it is to go viral on social media.

You have access to the following tools:
{tools}

Use this exact format for every step:

Question: the input question or task you must answer
Thought: think about what to do next
Action: the tool name to call (must be one of [{tool_names}])
Action Input: the input to pass to the tool
Observation: the result returned by the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to give a final answer
Final Answer: your complete, helpful response to the user

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")


class _GroqLLM(LLM):
    """
    A minimal LangChain-compatible wrapper around the Groq API.
    LangChain's ReAct agent needs an LLM object — this adapts our Groq client to that interface.
    """

    ## These tell LangChain which model and key to use
    ## Using the 70B model here because the agent needs to follow strict ReAct format
    ## The smaller 8B model loses track of the format and loops — 70B is much more reliable
    model_name: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        ## Create a fresh Groq client and send the prompt
        client = Groq(api_key=self.groq_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,  ## Zero temperature so the agent reasons deterministically
            stop=stop,
        )
        return response.choices[0].message.content


def build_agent() -> AgentExecutor:
    """
    Build and return the LangChain ReAct agent with all 6 tools attached.
    Uses Groq as the reasoning LLM (same model as our LLM explainability layer).
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    ## Wrap our Groq client so LangChain can use it as its reasoning engine
    llm = _GroqLLM(groq_api_key=api_key)
    ## Create the ReAct agent — it gets the tools list and the prompt template
    agent = create_react_agent(llm, TOOLS, REACT_PROMPT)
    ## AgentExecutor runs the agent loop: call tool → observe → repeat until Final Answer
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,       ## Print every Thought/Action/Observation step
        max_iterations=10,  ## Stop after 10 steps to avoid infinite loops
        handle_parsing_errors=True,
    )
