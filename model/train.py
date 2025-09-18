# model/train.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_tiny_dataset():
    pos = [
        "i love this",
        "amazing product",
        "great experience",
        "pleasant surprise",
        "absolutely fantastic",
        "highly recommend",
        "best ever",
        "works perfectly",
        "happy with the purchase",
        "five stars",
        "superb quality",
        "very satisfied",
        "brilliant and fast",
        "excellent service",
        "wonderful app",
        "awesome movie",
        "delightful",
        "exactly what i needed",
        "top notch",
        "worth it",
    ]
    neg = [
        "i hate this",
        "terrible product",
        "awful experience",
        "very disappointing",
        "absolutely horrible",
        "do not recommend",
        "worst ever",
        "broken on arrival",
        "waste of money",
        "one star",
        "poor quality",
        "not satisfied",
        "slow and buggy",
        "bad service",
        "horrible app",
        "boring movie",
        "frustrating",
        "not what i expected",
        "low quality",
        "regret buying",
    ]
    X = pos + neg
    y = ["positive"] * len(pos) + ["negative"] * len(neg)
    return X, y


def train_and_save(out_path: str | Path = "model/pipeline.pkl") -> dict[str, Any]:
    X, y = build_tiny_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # Meta info (now includes metrics)
    meta = {
        "model_name": "LogisticRegression + TFIDF",
        "version": "1.0.0",
        "trained_on": "tiny_sentiment_demo",
        "labels": sorted(list(set(y))),
        "metrics": {"accuracy": round(float(acc), 4), "f1_macro": round(float(f1_macro), 4)},
        "data": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": 0.25,
            "random_state": 42,
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, out_path)

    # Print required logs for the milestone
    print(f"Saved: {out_path}")
    print(f"Accuracy: {meta['metrics']['accuracy']:.4f}")
    print(f"F1-macro: {meta['metrics']['f1_macro']:.4f}")

    return {"ok": True, "path": str(out_path), "meta": meta}


if __name__ == "__main__":
    train_and_save()
