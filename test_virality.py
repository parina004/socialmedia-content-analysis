from pathlib import Path
import sys
sys.path.append('.')
from model_b.inference import predict

result = predict(
    video_path=Path(r"D:\Parina's folder\BITS Pilani\ALL SEMS\4th year\Project 1\datasets\testing-videos\deepfake_235 (1).mp4"),
    title='SHOCKING Discovery That Changes EVERYTHING! 😱🔥 #MustWatch',
    post_hour=18,
    post_day=4,  # Friday
    tag_count=15
)
print(f'Score: {result["virality_score"]}')
print(f'Label: {result["label"]}')
print(f'Probability: {result["probability"]}')
f = result['features']
print(f'\nKey features:')
print(f'  brisque: {f.get("brisque_score", 0):.2f}')
print(f'  vibrancy: {f.get("color_vibrancy", 0):.2f}')
print(f'  motion: {f.get("motion_intensity", 0):.2f}')
print(f'  rms_energy: {f.get("rms_energy", 0):.4f}')
print(f'  title_sentiment: {f.get("title_sentiment", 0):.2f}')
