import os
from huggingface_hub import snapshot_download

# If the model is gated, make sure you've accepted the license on HF
# and are logged in: `huggingface-cli login`, or set HF_TOKEN below.
HF_TOKEN = os.environ.get("HF_TOKEN")  # or paste a token string here

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
LOCAL_DIR = "models/flux2-klein-4b"

path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    token=HF_TOKEN,
    # keep it lean: skip files you don't need (e.g. alternate weight formats)
    # allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)

print(f"Downloaded {MODEL_ID} to: {path}")

