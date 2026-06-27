"""
Entry point for Hugging Face Spaces deployment.
HF Spaces expects the app to be in app.py and listen on port 7860.

Usage (in Space settings):
  SDK: Gradio → NO, use "Other" SDK
  App file: app.py

HF Spaces sets PORT=7860 automatically.
"""

from app import app   # reuse the Flask app from app.py

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
