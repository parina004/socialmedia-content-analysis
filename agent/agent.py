## Builds and runs the LangChain ReAct agent with all 6 tools.                                               
## The agent reasons step by step
## decides which tools to call and in what order based on intermediate results, rather than following a fixed pipeline.                              
                                                                                                            
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from tools import (
    run_synthetic_detection,
    run_virality_prediction,
    search_knowledge_base, 
    fetch_trending_hashtags,
    generate_forensic_report,
    generate_virality_report,
)

## All 6 tools the agent can choose from
TOOLS = [
    run_synthetic_detection,
    run_virality_prediction,
    search_knowledge_base,
    fetch_trending_hashtags,
    generate_forensic_report,
    generate_virality_report,
]

## ReAct prompt template — this is the instruction manual the agent follows.
## It tells the agent the exact format to use for Thought/Action/Observation.
## {tools} and {tool_names} are filled in automatically from the TOOLS list.
## {input} is the user's request. {agent_scratchpad} holds the reasoning so far.
REACT_PROMPT = PromptTemplate.from_template("""
You are an AI analysis agent for social media content. You have access to tools
that detect synthetic media (deepfakes/AI-generated video) and predict virality.

Always follow this exact format for every step:

Thought: think about what you need to do next
Action: the tool name to call (must be one of: {tool_names})
Action Input: the input to pass to the tool
Observation: the result returned by the tool

Repeat Thought/Action/Action Input/Observation as many times as needed.
When you have completed all analysis and written both reports, end with:

Thought: I have completed the full analysis.
Final Answer: [your complete summary with both the forensic verdict and virality analysis]

Available tools:
{tools}

Request: {input}

{agent_scratchpad}
""")


def build_agent() -> AgentExecutor:
    llm = ChatGroq(model = "llama-3.3-70b-versatile",temperature = 0)

    agent = create_react_agent(
        llm    = llm,
        tools  = TOOLS,
        prompt = REACT_PROMPT,
    )

    return AgentExecutor(
        agent               = agent,
        tools               = TOOLS,
        verbose             = True,    ## prints every Thought/Action/Observation
        handle_parsing_errors = True,  ## recovers if LLM formats output wrong
        max_iterations      = 20,      ## prevents infinite loops
    )


def run_analysis(
    video_path:    str,
    title:         str,
    post_hour:     int,
    post_day:      int,
    tag_count:     int,
    user_caption:  str = "",
    user_hashtags: str = "",
    topic:         str = "general",
) -> str:
    executor = build_agent()

    request = (
        f"Analyse this video completely:\n"
        f"- Video path: {video_path}\n"
        f"- Title: {title}\n"
        f"- Post hour: {post_hour}, Post day: {post_day}, Tag count: {tag_count}\n"
        f"- User caption: {user_caption or 'Not provided'}\n"
        f"- User hashtags: {user_hashtags or 'Not provided'}\n"
        f"- Topic: {topic}\n\n"
        f"You must complete ALL of the following steps before giving a Final Answer:\n"
        f"1. Run synthetic detection on the video\n"
        f"2. Run virality prediction on the video\n"
        f"3. Fetch trending hashtags for the topic\n"
        f"4. Generate the forensic report using the detection result from step 1\n"
        f"5. Generate the virality report using the prediction result from step 2\n"
        f"Do not give a Final Answer until all 5 steps are complete.\n"
    )

    result = executor.invoke({"input": request})
    return result["output"]


if __name__ == "__main__":
    output = run_analysis(
        video_path    = r"D:\todelete\ai.mp4",
        title         = "Test Video Title",
        post_hour     = 15,
        post_day      = 2,
        tag_count     = 5,
        user_caption  = "Check out this amazing video!",
        user_hashtags = "#test #video",
        topic         = "general",
    )
    print("\n=== FINAL OUTPUT ===")
    print(output)
