# PromptAttack — MVP

**PromptAttack (MVP)** is a lightweight prompt-jailbreak detector: SBERT embeddings + Logistic Regression trained on toxicity and in-the-wild jailbreak prompt datasets.
This repo contains training and inference scripts to detect suspicious/jailbreak prompts before they reach an LLM.

## Quick start (local)
```bash
# recommended: work inside the project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-only torch install (faster/smaller) then minimal deps
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-cache-dir sentence-transformers datasets scikit-learn joblib tqdm

# train a quick model
python promptattack/train_detector.py --max-examples 1000 --embedder all-mpnet-base-v2 --outname sbert_lr_detector_quick

# run inference
python promptattack/infer.py --model models/sbert_lr_detector_quick.joblib --file examples/test_prompts.txt --threshold 0.5
