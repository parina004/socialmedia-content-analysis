"""
llm/test_llm.py
Run with: uv run llm/test_llm.py
Tests the full LLM pipeline in order: prompts.py first, then llm_client.py.
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from prompts import format_synthetic_prompt, format_virality_prompt
from llm_client import generate

## Step 1: Build a fake synthetic detection result and format it into a prompt
print("=" * 60)
print("TEST 1 — Synthetic Detection Prompt")
print("=" * 60)

synthetic_prompt = format_synthetic_prompt(
    label="Deepfake",
    confidence=91.3,
    efficientnet_score=0.8821,
    forensic_score=0.9104,
    face_score=0.7643,
    rag_context="No RAG context — test run only.",
)

print("\nFormatted prompt:\n")
print(synthetic_prompt)

## Step 2: Send that prompt to the LLM and print the reply
print("\nSending to LLM …\n")
reply1 = generate(synthetic_prompt)
print("LLM reply:\n", reply1)


## Step 3: Build a fake virality prediction result and format it into a prompt
print("\n" + "=" * 60)
print("TEST 2 — Virality Analysis Prompt")
print("=" * 60)

virality_prompt = format_virality_prompt(
    virality_label="Viral",
    viral_probability=78.5,
    engagement_percentile=82,
    brisque=24.3,
    vibrancy=0.72,
    motion=0.54,
    face_ratio=0.65,
    tempo=128.0,
    rms_energy=0.0431,
    speech_ratio=0.61,
    title_sentiment=0.45,
    title_length=52,
    tag_count=8,
    upload_hour=18,
    upload_day="Friday",
    user_caption="Check out my latest travel vlog!",
    user_hashtags="#travel #vlog #explore",
    rag_context="No RAG context — test run only.",
)

print("\nFormatted prompt:\n")
print(virality_prompt)

## Step 4: Send that prompt to the LLM and print the reply
print("\nSending to LLM …\n")
reply2 = generate(virality_prompt)
print("LLM reply:\n", reply2)
