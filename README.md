# PromptAttack

**PromptAttack** is a lightweight **prompt-jailbreak detector** for LLMs.

It uses **SBERT sentence embeddings** (`all-mpnet-base-v2`) plus a **Logistic Regression** classifier trained on open safety datasets (RealToxicityPrompts + in-the-wild jailbreak-style prompts) to estimate how likely a text prompt is to be harmful or jailbreak-like *before* you send it to a model.

This repo is intentionally small and focused: it shows the full pipeline end-to-end (data loading → embeddings → training → saved artifact → CLI inference).

## Quick start (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-cache-dir sentence-transformers datasets scikit-learn joblib tqdm

python promptattack/train_detector.py --max-examples 1000 --embedder all-mpnet-base-v2 --outname sbert_lr_detector_quick
python promptattack/infer.py --model models/sbert_lr_detector_quick.joblib --file examples/test_prompts.txt --threshold 0.5
