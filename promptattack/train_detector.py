#!/usr/bin/env python3
"""
train_detector.py
MVP: download safety/jailbreak-style datasets, train a SBERT+LogisticRegression detector,
evaluate, and save model artifacts (scaler + classifier + embedder name).
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBED = "all-mpnet-base-v2"  # good SBERT compromise


def load_wildjailbreak_subset(limit_per_class=20000):
    """
    Try to load allenai/wildjailbreak (gated dataset).
    If not accessible, returns empty lists and prints a warning.
    """
    texts = []
    labels = []
    try:
        ds = load_dataset("allenai/wildjailbreak", split="train")
        for ex in ds:
            txt = ex.get("prompt") or ex.get("input") or ex.get("text") or ex.get("content")
            if txt is None:
                continue
            lab = None
            if "label" in ex and ex["label"] is not None:
                lab = ex["label"]
            elif "is_harmful" in ex:
                lab = int(ex["is_harmful"])
            if lab is None:
                continue
            texts.append(str(txt).strip())
            labels.append(int(lab))
            if len(texts) >= limit_per_class * 2:
                break
        print(f"[wildjailbreak] loaded {len(texts)} examples")
    except Exception as e:
        print("Warning: could not load allenai/wildjailbreak:", e)
    return texts, labels


def load_in_the_wild(limit_per_split=2500):
    """
    Load TrustAIRLab/in-the-wild-jailbreak-prompts with explicit configs.
    'jailbreak_*' splits are labeled 1, 'regular_*' splits are labeled 0.
    """
    texts = []
    labels = []
    try:
        jailbreak_cfgs = ["jailbreak_2023_05_07", "jailbreak_2023_12_25"]
        regular_cfgs = ["regular_2023_05_07", "regular_2023_12_25"]

        # jailbreak (label 1)
        for cfg in jailbreak_cfgs:
            ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", cfg, split="train")
            for ex in ds:
                txt = ex.get("prompt") or ex.get("text") or ex.get("input") or ex.get("content")
                if txt is None:
                    continue
                s = str(txt).strip()
                if not s:
                    continue
                texts.append(s)
                labels.append(1)
                if len(texts) >= limit_per_split:
                    break
            if len(texts) >= limit_per_split:
                break

        # regular (label 0)
        for cfg in regular_cfgs:
            ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", cfg, split="train")
            for ex in ds:
                txt = ex.get("prompt") or ex.get("text") or ex.get("input") or ex.get("content")
                if txt is None:
                    continue
                s = str(txt).strip()
                if not s:
                    continue
                texts.append(s)
                labels.append(0)
                if len(texts) >= 2 * limit_per_split:
                    break
            if len(texts) >= 2 * limit_per_split:
                break

        print(f"[in-the-wild] loaded {len(texts)} examples")
    except Exception as e:
        print("Warning: could not load TrustAIRLab/in-the-wild-jailbreak-prompts:", e)

    return texts, labels


def load_real_toxicity_prompts(limit=20000, toxic_thr=0.5, safe_thr=0.2):
    """
    Load allenai/real-toxicity-prompts, using toxicity scores as labels.
    - label 1: prompt.toxicity >= toxic_thr
    - label 0: prompt.toxicity <= safe_thr
    Prompts in the middle region are skipped.
    """
    texts = []
    labels = []
    try:
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")
        for ex in ds:
            prompt = ex.get("prompt")
            if not isinstance(prompt, dict):
                continue
            text = prompt.get("text")
            tox = prompt.get("toxicity")
            if text is None or tox is None:
                continue
            s = str(text).strip()
            if not s:
                continue

            # label based on toxicity
            try:
                tval = float(tox)
            except Exception:
                continue

            if tval >= toxic_thr:
                texts.append(s)
                labels.append(1)
            elif tval <= safe_thr:
                texts.append(s)
                labels.append(0)

            if len(texts) >= limit:
                break
        print(f"[real-toxicity-prompts] loaded {len(texts)} examples")
    except Exception as e:
        print("Warning: could not load allenai/real-toxicity-prompts:", e)
    return texts, labels


def assemble_dataset(max_examples=20000):
    """
    Assemble a dataset from available sources.
    Priority: real-toxicity-prompts (open), plus in-the-wild, plus wildjailbreak if accessible.
    """
    x = []
    y = []

    # 1) real-toxicity-prompts (open, non-gated)
    tx_rt, ty_rt = load_real_toxicity_prompts(limit=max_examples)
    x.extend(tx_rt)
    y.extend(ty_rt)

    # 2) in-the-wild jailbreak prompts (if available)
    tx_itw, ty_itw = load_in_the_wild(limit_per_split=max_examples // 4)
    x.extend(tx_itw)
    y.extend(ty_itw)

    # 3) wildjailbreak (gated; may give 0 examples)
    tx_wb, ty_wb = load_wildjailbreak_subset(limit_per_class=max_examples // 4)
    x.extend(tx_wb)
    y.extend(ty_wb)

    print(f"[assemble] raw loaded: {len(x)} examples")

    if len(x) == 0:
        raise RuntimeError(
            "No dataset loaded. Ensure internet and HF datasets are available, "
            "or adjust the data loaders to use a different open dataset."
        )

    # dedupe and basic cleanup
    seen = set()
    X2 = []
    Y2 = []
    for t, l in zip(x, y):
        s = t.strip()
        if len(s) < 3:
            continue
        if s in seen:
            continue
        seen.add(s)
        X2.append(s)
        Y2.append(int(l))

    print(f"[assemble] after dedupe/cleanup: {len(X2)} examples")
    return X2, Y2


def embed_texts(texts, model_name):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return emb, model


def train_and_save(X_emb, y, embedder_name, out_prefix="detector"):
    X = np.asarray(X_emb)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_proba))
    except Exception:
        print("ROC AUC: could not compute (maybe single-class in test split).")

    print(classification_report(y_test, y_pred))

    artifact = {"scaler": scaler, "clf": clf, "embedder": embedder_name}
    out_path = OUTPUT_DIR / f"{out_prefix}.joblib"
    joblib.dump(artifact, out_path)
    print("Saved model to", out_path)
    return clf, scaler


def main(args):
    print("Assembling dataset... (this may take a while)")
    X_texts, y = assemble_dataset(max_examples=args.max_examples)
    print(f"Loaded {len(X_texts)} prompts, {sum(y)} labeled as attack/harmful")

    print("Embedding with", args.embedder)
    X_emb, _ = embed_texts(X_texts, args.embedder)

    print("Training detector...")
    train_and_save(X_emb, y, args.embedder, out_prefix=args.outname)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-examples", type=int, default=20000, help="max examples to assemble")
    p.add_argument("--embedder", type=str, default=DEFAULT_EMBED, help="SBERT model name")
    p.add_argument("--outname", type=str, default="sbert_lr_detector", help="output model name")
    args = p.parse_args()
    main(args)
