#!/usr/bin/env python3
"""
train_detector.py
MVP: download jailbreak datasets, train a SBERT+LogisticRegression detector,
evaluate, and save model artifacts (scaler + classifier + embedder name).
"""
import os
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBED = "all-mpnet-base-v2"  # good SBERT compromise

def load_wildjailbreak_subset(limit_per_class=20000):
    texts = []
    labels = []
    try:
        ds = load_dataset("allenai/wildjailbreak", split="train")
        for ex in ds:
            txt = ex.get("prompt") or ex.get("input") or ex.get("text") or ex.get("content")
            if txt is None: 
                continue
            lab = None
            # different dataset versions may use different fields
            if "label" in ex and ex["label"] is not None:
                lab = ex["label"]
            elif "is_harmful" in ex:
                lab = int(ex["is_harmful"])
            # Skip if label still unknown
            if lab is None:
                continue
            texts.append(str(txt).strip())
            labels.append(int(lab))
            # early stop guard
            if len(texts) >= limit_per_class * 2:
                break
    except Exception as e:
        print("Warning: could not load allenai/wildjailbreak:", e)
    return texts, labels

def load_in_the_wild(limit=10000):
    texts = []
    labels = []
    try:
        ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", split="train")
        for ex in ds:
            txt = ex.get("prompt") or ex.get("text") or ex.get("input") or ex.get("content")
            if txt is None:
                continue
            tag = ex.get("label") or ex.get("category") or ex.get("type") or ex.get("split")
            if tag is None:
                if "jailbreak" in ex:
                    lab = int(ex["jailbreak"])
                else:
                    continue
            else:
                t = str(tag).lower()
                if "jail" in t or "attack" in t or "adv" in t:
                    lab = 1
                elif "regular" in t or "benign" in t or "clean" in t:
                    lab = 0
                else:
                    continue
            texts.append(str(txt).strip())
            labels.append(int(lab))
            if len(texts) >= limit:
                break
    except Exception as e:
        print("Warning: could not load TrustAIRLab/in-the-wild-jailbreak-prompts:", e)
    return texts, labels

def assemble_dataset(max_examples=20000):
    x = []
    y = []
    tx, ty = load_wildjailbreak_subset(limit_per_class=max_examples//2)
    x.extend(tx); y.extend(ty)
    tx2, ty2 = load_in_the_wild(limit=max_examples//2)
    x.extend(tx2); y.extend(ty2)
    if len(x) == 0:
        raise RuntimeError("No dataset loaded. Ensure internet and HF datasets available.")
    # dedupe and basic cleanup
    seen = set(); X2=[]; Y2=[]
    for t, l in zip(x, y):
        s = t.strip()
        if len(s) < 3: continue
        if s in seen: continue
        seen.add(s)
        X2.append(s); Y2.append(l)
    return X2, Y2

def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True), model

def train_and_save(X_emb, y, embedder_name, out_prefix="detector"):
    X = np.asarray(X_emb)
    y = np.asarray(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:,1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        print("ROC AUC: could not compute (maybe single-class in test split).")
    print(classification_report(y_test, y_pred))
    joblib.dump({"scaler": scaler, "clf": clf, "embedder": embedder_name}, OUTPUT_DIR / f"{out_prefix}.joblib")
    print("Saved model to", OUTPUT_DIR / f"{out_prefix}.joblib")
    return clf, scaler

def main(args):
    print("Assembling dataset... (this may take a while)")
    X_texts, y = assemble_dataset(max_examples=args.max_examples)
    print(f"Loaded {len(X_texts)} prompts, {sum(y)} jailbreaks labeled")
    print("Embedding with", args.embedder)
    X_emb, model = embed_texts(X_texts, args.embedder)
    print("Training detector...")
    clf, scaler = train_and_save(X_emb, y, args.embedder, out_prefix=args.outname)
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-examples", type=int, default=20000, help="max examples to assemble")
    p.add_argument("--embedder", type=str, default=DEFAULT_EMBED, help="SBERT model name")
    p.add_argument("--outname", type=str, default="sbert_lr_detector", help="output model name")
    args = p.parse_args()
    main(args)
