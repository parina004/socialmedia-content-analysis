import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agent.agent import run_analysis

if __name__ == "__main__":
    output = run_analysis(
        video_path    = r"C:\Users\parin\Downloads\testingvideo.mp4",
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

