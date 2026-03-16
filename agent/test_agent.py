"""
agent/test_agent.py
Run with: uv run agent/test_agent.py <path_to_video>
Runs the agent end-to-end on a given video to verify all 6 tools work correctly.
"""

import sys
import logging
from pathlib import Path

## Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from agent.agent import build_agent

if len(sys.argv) < 2:
    print("Usage: uv run agent/test_agent.py <path_to_video>")
    sys.exit(1)

VIDEO = sys.argv[1]

## Build the agent
agent = build_agent()

## Ask it to do a full analysis — it will decide which tools to call and in what order
result = agent.invoke({
    "input": (
        f"Please analyze the video at '{VIDEO}'. "
        "Check if it is synthetic media, predict its virality, "
        "and give me a full report with recommendations. "
        "My planned caption is 'Check out this amazing clip!' "
        "and my hashtags are #viral #trending #explore."
    )
})

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(result["output"])
