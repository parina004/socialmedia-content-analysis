FROM python:3.10-slim

# System libraries required by OpenCV, MediaPipe, LightGBM, and ffmpeg (audio extraction)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgstreamer-plugins-base1.0-0 \
    libgomp1 \ 
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model files from HF Model repo
RUN python -c "\
import shutil; \
from huggingface_hub import hf_hub_download; \
shutil.copy(hf_hub_download(repo_id='parina13/synthsenses-models', filename='forensic_cnn.pth', repo_type='model'), 'model_a/forensic_cnn.pth'); \
shutil.copy(hf_hub_download(repo_id='parina13/synthsenses-models', filename='model.pkl', repo_type='model'), 'model_b/model.pkl'); \
print('Models downloaded')"

# HF Spaces Docker uses port 7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]