#!/usr/bin/env python3
"""
infer.py - CLI inference for PromptAttack.

Usage examples:
  python promptattack/infer.py --model models/sbert_lr_detector_quick.joblib --prompt "hello"
  python promptattack/infer.py --model models/sbert_lr_detector_quick.joblib --file examples/test_prompts.txt
"""
import argparse
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

# ---------- Heuristics ----------
def short_alpha_heuristic(text):
    s = (text or "").strip().lower()
    if not s:
        return 0.0, "EMPTY", "Empty input."
    tokens = s.split()
    if len(tokens) <= 2 and all(tok.isalpha() for tok in tokens):
        return 0.01, "SAFE", "Heuristic: very short alphabetic prompt (likely greeting)."
    return None, None, None

def danger_phrase_heuristic(text):
    danger_patterns = [
        "make a bomb", "explosive", "weapon", "bypass authentication",
        "disable safety", "hack", "jailbreak", "ignore previous instructions",
        "illegal", "harm", "kill", "attack", "from household materials"
    ]
    s = (text or "").lower()
    for pat in danger_patterns:
        if pat in s:
            return 0.99, "JAILBREAK", "Heuristic: matched danger phrase '{}'".format(pat)
    return None, None, None
# ---------------------------------

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

    model_path = Path(args.model)
    if not model_path.exists():
        print("Model artifact not found:", args.model)
        return

    scaler, clf, embedder_name = load_artifact(args.model)

    if args.prompt:
        texts = [args.prompt.strip()]
    elif args.file:
        pth = Path(args.file)
        if not pth.exists():
            print("File not found:", args.file)
            return
        with open(pth, "r", encoding="utf8") as f:
            texts = [l.strip() for l in f if l.strip()]
    else:
        print("No prompt or file provided. Use --prompt or --file.")
        return

    probs = score_texts(texts, scaler, clf, embedder_name)

    for t, p_prob in zip(texts, probs):
        # Heuristic: short benign greeting
        h_prob, h_verdict, h_note = short_alpha_heuristic(t)
        if h_verdict is not None:
            prob, verdict, note = h_prob, h_verdict, h_note
        else:
            # Heuristic: obvious dangerous phrases
            h_prob2, h_verdict2, h_note2 = danger_phrase_heuristic(t)
            if h_verdict2 is not None:
                prob, verdict, note = h_prob2, h_verdict2, h_note2
            else:
                prob = float(p_prob)
                verdict = "JAILBREAK" if prob >= args.threshold else "SAFE"
                note = "Threshold={:.2f}, Probability={:.3f}".format(args.threshold, prob)

        print("="*80)
        print("Verdict: {}".format(verdict))
        print("Probability: {:.6f}".format(prob))
        print("Notes: {}".format(note))
        print("Prompt: {}\n".format(t))

if __name__ == "__main__":
    main()
