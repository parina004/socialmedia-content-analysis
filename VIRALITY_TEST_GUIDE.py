"""
OPTIMAL VALUES FOR TESTING VIRALITY
Based on model feature importances and typical viral content
"""

# Hardcoded metrics (can't change these - they're averages)
# like_to_view_ratio: 0.04 (4%)
# comment_to_view_ratio: 0.005 (0.5%)

# VIDEO CONTENT (extracted from file - must use a high-quality video):
# - brisque_score: Lower is better (< 30 = excellent quality)
# - color_vibrancy: Higher is better (> 150)
# - motion_intensity: Moderate (1-3 for dynamic content)
# - rms_energy: Higher is better (> 0.05 for energetic audio)
# - face_presence_ratio: Higher for talking-head content (0.8-1.0)

# USER INPUTS (what you control):
OPTIMAL_INPUTS = {
    "title": "SHOCKING Discovery That Changes Everything! 😱🔥 #MustWatch",  # Emotional, question, emojis
    "post_hour": 18,  # 6 PM - prime evening time
    "post_day": 4,    # Friday (0=Monday, 4=Friday)
    "tag_count": 15   # More tags = better discoverability
}

# Title tips:
# - Use emotional words: "shocking", "incredible", "must-see"
# - Add question: "Will this SHOCK you?"
# - Include numbers: "Top 5 Secrets"
# - Add emojis: 😱🔥💯
# - Keep length 50-70 chars for optimal sentiment

# Best posting times:
# - 18:00 (6 PM) - After work/school
# - 20:00 (8 PM) - Prime time
# - 12:00 (Noon) - Lunch break

# Best posting days:
# - Thursday (3)
# - Friday (4)
# - Sunday (6)
