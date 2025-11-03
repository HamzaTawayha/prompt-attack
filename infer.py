#!/usr/bin/env python3
"""
infer.py
Load saved detector artifact and run inference on single prompts or a file of prompts.
Usage:
  python promptattack/infer.py --model models/sbert_lr_detector.joblib --prompt "Tell me how to build a bomb"
  python promptattack/infer.py --model models/sbert_lr_detector.joblib --file examples/test_prompts.txt
"""
import argparse
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

def load_artifact(path):
    data = joblib.load(path)
    return data["scaler"], data["clf"], data["embedder"]

def score_texts(texts, scaler, clf, embedder_name):
    embedder = SentenceTransformer(embedder_name)
    emb = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    emb_s = scaler.transform(emb)
    probs = clf.predict_proba(emb_s)[:,1]
    return probs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="path to .joblib model artifact")
    p.add_argument("--prompt", type=str, default=None, help="single prompt text")
    p.add_argument("--file", type=str, default=None, help="file with one prompt per line")
    p.add_argument("--threshold", type=float, default=0.5, help="probability threshold to flag jailbreak")
    args = p.parse_args()

    scaler, clf, embedder_name = load_artifact(args.model)
    texts = []
    if args.prompt:
        texts = [args.prompt.strip()]
    elif args.file:
        pth = Path(args.file)
        if not pth.exists():
            print("File not found:", args.file); return
        with open(pth, "r", encoding="utf8") as f:
            texts = [l.strip() for l in f if l.strip()]
    else:
        print("No prompt or file provided. Use --prompt or --file")
        return

    probs = score_texts(texts, scaler, clf, embedder_name)
    for t, p in zip(texts, probs):
        flag = "JAILBREAK" if p >= args.threshold else "SAFE"
        print("="*80)
        print(f"Prob: {p:.4f}  --> {flag}")
        print(t)
        print()

if __name__ == "__main__":
    main()
