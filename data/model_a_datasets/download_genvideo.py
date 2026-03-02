import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")

snapshot_download(
    repo_id="faridlab/deepaction_v1",
    repo_type="dataset",
    local_dir="data/model_a_datasets/ai_datasets/GenVideo/deepaction",
    token=token
)

print("Download complete!")