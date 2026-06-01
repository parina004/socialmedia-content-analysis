"""
llm/llm_client.py
Single entry-point for generating LLM explanations via Groq.
This file never sees raw video — it only receives formatted prompt strings from prompts.py.
"""

from __future__ import annotations

import os
import logging

## Load the .env file so GROQ_API_KEY is available without manually exporting it in terminal
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

## The Groq cloud model — LLaMA 3.1 8B, fast and free tier
GROQ_MODEL = "llama-3.1-8b-instant"

## How many tokens the LLM can write in its reply — 512 is enough for a short explanation
MAX_TOKENS = 512


def _call_groq(prompt: str) -> str:
    from groq import Groq  # type: ignore

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Sign up free at console.groq.com, generate a key, and add it to your .env file."
        )

    client = Groq(api_key=api_key)

    ## Tell the model what role it's playing before giving it the main prompt
    system_message = (
        "You are an expert AI assistant that explains synthetic media detection "
        "and social media virality prediction results to users in clear, professional prose."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,  ## A little creative but still factual
    )

    return response.choices[0].message.content.strip()


def generate(prompt: str) -> str:
    """
    Send a formatted prompt to Groq and return the plain-text explanation.
    """
    try:
        logger.info("Attempting Groq (%s) …", GROQ_MODEL)
        result = _call_groq(prompt)
        logger.info("Groq succeeded.")
        return result

    except Exception as groq_err:
        logger.error("Groq failed: %s", groq_err)
        raise RuntimeError(
            f"LLM generation failed.\n"
            f"  Groq error: {groq_err}\n"
            f"  Ensure GROQ_API_KEY is set in your .env file (free at console.groq.com)."
        ) from groq_err


## Run this file directly to check everything is wired up correctly
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ## A tiny test prompt — just verifies the pipeline connects end to end
    test_prompt = (
        "In one sentence, confirm that the LLM explainability layer is working correctly."
    )

    print("Testing LLM client …\n")
    try:
        reply = generate(test_prompt)
        print("Reply:", reply)
    except RuntimeError as e:
        print("ERROR:", e)
        sys.exit(1)
