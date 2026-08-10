# Image Captioning using Flickr8k Dataset

An end-to-end image captioning pipeline using BLIP (Salesforce/blip-image-captioning-base), with explainability (XAI) analysis and a full MLOps pipeline (versioning, tracking, CI/CD, and a demo app).

## Project Overview

This project explores zero-shot image captioning on the Flickr8k dataset, evaluates baseline performance, investigates *why* the model succeeds or fails using explainability techniques, and wraps the pipeline in production-style MLOps tooling.

## Architecture
## Dataset
- **Flickr8k**: 8,091 images, 40,455 captions (5 per image)
- Preprocessing: NLTK (tokenization, stopword removal), OpenCV (resize to 224x224, normalization)

## Baseline Model
- **Model**: Salesforce/blip-image-captioning-base (BLIP), used zero-shot (no fine-tuning)
- **Final config**: Beam Search, num_beams=5, max_new_tokens=30
- **Evaluation (200 images)**: BLEU=0.1919, ROUGE-1=0.5556, ROUGE-2=0.3198, ROUGE-L=0.5337
- Config comparison (greedy vs. beam search) tracked and compared in MLflow

## Explainability (XAI)
- Grad-CAM-style attention heatmaps (ViT attention rollout) on the vision encoder
- Word-by-word cross-attention alignment (which image region influenced each generated word)
- LIME-inspired occlusion testing (patch-based perturbation analysis)
- Key finding: occluding a distracting background patch changed a caption from "a sandy beach" to "a woman walking on a beach," showing that background clutter can suppress correct subject detection

## MLOps Pipeline
- **Modular code**: `Week4/src/` (data_loader.py, model.py, inference.py, xai.py)
- **DVC**: versions the curated dataset and model artifact config
- **MLflow**: experiment tracking + Model Registry (`BLIP_Image_Captioning_Baseline`)
- **pytest**: unit tests for data loading and inference (`Week4/tests/test_pipeline.py`)
- **GitHub Actions CI/CD**: automated linting (flake8/black), testing (pytest), and Docker image build on every push to main
- **Docker**: containerized app (`Week4/Dockerfile`)
- **Gradio app**: upload an image, get a BLIP caption + attention heatmap overlay side-by-side

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Aiman-Malik553/Python-Internship.git
cd Python-Internship/Week4

# Install dependencies
pip install -r requirements.txt

# Run the demo app
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.

## Run Tests

```bash
cd Week4
python -m pytest tests/test_pipeline.py -v
```

## Known Limitations
- Zero-shot BLIP struggles with uncommon object names (e.g., "Frisbee" was garbled in output)
- Captions are generally shorter and more generic than human-written captions
- Performance drops on complex/dynamic scenes (e.g., a horse race was captioned as "a clear blue sky")
- Public deployment (Hugging Face Spaces / Render) requires payment card verification for Python-based hosting under current free-tier policies; the app is fully functional locally and via Docker, and can be deployed in minutes to any verified account

## Future Work
- Fine-tune BLIP on the Flickr8k training set (freeze vision encoder, train text decoder only)
- Expand evaluation to the full validation set
- Explore ONNX export for faster inference
- Deploy to a verified cloud account (Hugging Face Spaces / Render / AWS ECS)

## Tech Stack
Python, PyTorch, HuggingFace Transformers, MLflow, DVC, Docker, GitHub Actions, Gradio, pytest, NLTK, OpenCV

## Author
Aiman Malik
