"""Test Pyannote model download with authentication."""

import os
import sys

try:
    from pyannote.audio import Pipeline
    from huggingface_hub import login
except ImportError:
    print("Installing pyannote...")
    os.system("pip install pyannote.audio")
    from pyannote.audio import Pipeline
    from huggingface_hub import login

# Get token from environment
HF_TOKEN = os.environ.get("PYANNOTE_AUTH_TOKEN")
if not HF_TOKEN:
    print("No HF_TOKEN found in environment")
    sys.exit(1)

print(f"Token: {HF_TOKEN[:10]}...")

# Login with token
try:
    login(token=HF_TOKEN)
    print("Successfully logged in to Hugging Face")
except Exception as e:
    print(f"Login failed: {e}")
    sys.exit(1)

# Try to download the pipeline
print("Downloading Pyannote pipeline...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    print("Pipeline downloaded successfully!")
    print(f"Pipeline type: {type(pipeline)}")
except Exception as e:
    print(f"Download failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
