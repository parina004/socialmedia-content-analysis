"""
llm/llm_client.py
Single entry-point for generating LLM explanations.
Tries Ollama (local, free) first. If that fails, falls back to Groq (cloud, free tier).
This file never sees raw video — it only receives formatted prompt strings from prompts.py.
"""

from __future__ import annotations

import os
import logging

## Load the .env file so GROQ_API_KEY is available without manually exporting it in terminal
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

## The local Ollama model name — must be pulled first with: ollama pull llama3
OLLAMA_MODEL = "llama3"

## The Groq cloud model — LLaMA 3.1 8B, fast and free tier, same size as our local Ollama model
GROQ_MODEL = "llama-3.1-8b-instant"

## How many tokens the LLM can write in its reply — 512 is enough for a short explanation
MAX_TOKENS = 512


def _call_ollama(prompt: str) -> str:
    ## Import ollama inside the function so the rest of the file still works if ollama isn't installed
    import ollama  # type: ignore

    ## Send the prompt as a user message and get the assistant's reply back
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": MAX_TOKENS},
    )
    ## The actual reply text is buried inside response["message"]["content"]
    return response["message"]["content"].strip()


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
    Send a formatted prompt to the LLM and return the plain-text explanation.
    Tries local Ollama first, then falls back to Groq if Ollama is unavailable.
    """
    ## First try: local Ollama — no internet or API key needed
    try:
        logger.info("Attempting Ollama (%s) …", OLLAMA_MODEL)
        result = _call_ollama(prompt)
        logger.info("Ollama succeeded.")
        return result

    except Exception as ollama_err:
        ## Ollama is not running or the model hasn't been pulled — try Groq instead
        logger.warning(
            "Ollama unavailable (%s). Falling back to Groq %s.",
            ollama_err,
            GROQ_MODEL,
        )

    ## Second try: Groq cloud — needs GROQ_API_KEY in .env
    try:
        logger.info("Attempting Groq (%s) …", GROQ_MODEL)
        result = _call_groq(prompt)
        logger.info("Groq succeeded.")
        return result

    except Exception as groq_err:
        ## Both options failed — raise a clear error so the user knows what to fix
        logger.error("Groq also failed: %s", groq_err)
        raise RuntimeError(
            f"LLM generation failed.\n"
            f"  Ollama error : check that `ollama serve` is running and `ollama pull llama3` was done.\n"
            f"  Groq error   : {groq_err}\n"
            f"  Set GROQ_API_KEY in your .env file (free at console.groq.com)."
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
