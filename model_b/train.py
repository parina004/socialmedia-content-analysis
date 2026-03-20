## Model B — Virality Prediction: Training Script

## This script loads the pre-extracted features.csv, tunes a LightGBM
## classifier using Optuna (50 trials, 5-fold stratified CV on ROC-AUC),
## then trains the final model on the full training set and evaluates it.

## The dataset is imbalanced (viral ~33%, not-viral ~67%), so we compute
## scale_pos_weight from the training labels and pass it to every trial.

## Outputs saved to model_b/:
##   model.pkl              - the trained LGBMClassifier
##   metrics.json           - all evaluation numbers as a JSON dict
##   metrics.txt            - full human-readable report
##   feature_importance.png - bar chart showing which features matter most

## Run with:  uv run model_b/train.py

import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   ## headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)  ## suppress per-trial noise


## paths
ROOT         = Path(__file__).parent.parent
INPUT        = ROOT / "data" / "model_b_datasets" / "csv" / "features.csv"
OUT_DIR      = ROOT / "model_b"
MODEL_PATH   = OUT_DIR / "model.pkl"
METRICS_JSON = OUT_DIR / "metrics.json"
METRICS_TXT  = OUT_DIR / "metrics.txt"
FI_PLOT      = OUT_DIR / "feature_importance.png"


## settings
SEED       = 42
TEST_RATIO = 0.20
N_TRIALS   = 50
N_FOLDS    = 5


## the 22 features the model trains on (columns 4–25 in features.csv)
FEATURE_COLS = [
    ## visual (7)
    "brisque_score", "color_vibrancy", "motion_intensity",
    "face_presence_ratio", "face_emotion_joy", "face_emotion_surprise",
    "thumbnail_brightness",
    ## audio (5)
    "tempo_bpm", "rms_energy", "speech_ratio", "zero_crossing_rate", "beat_strength",
    ## metadata (8)
    "title_sentiment", "title_length", "title_has_question", "title_has_number",
    "description_length", "tag_count", "upload_hour", "upload_day",
    ## engagement ratios (2) — set to dataset mean at inference for new videos
    "like_to_view_ratio", "comment_to_view_ratio",
]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(INPUT)
    log.info(f"Loaded {len(df)} rows  ·  {INPUT.name}")
    log.info(f"  label counts : {df['label'].value_counts().to_dict()}")

    before = len(df)
    df     = df.dropna(subset=FEATURE_COLS + ["label"])
    if len(df) < before:
        log.warning(f"  dropped {before - len(df)} rows with missing values")

    X    = df[FEATURE_COLS].astype(np.float32)   ## keep as DataFrame so LightGBM retains feature names
    y    = (df["label"] == "viral").astype(int).values
    ids  = df["video_id"].values
    return X, y, ids


def make_objective(X_train: np.ndarray, y_train: np.ndarray, spw: float):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":  spw,
            "objective":         "binary",
            "verbosity":         -1,
            "random_state":      SEED,
            "n_jobs":            -1,
        }
        scores = cross_val_score(
            LGBMClassifier(**params),
            X_train, y_train,
            cv=cv,
            scoring="roc_auc",
        )
        return float(scores.mean())

    return objective


def evaluate(model: LGBMClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":               float(accuracy_score(y_test, y_pred)),
        "precision":              float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":                 float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                     float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":                float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(
            y_test, y_pred, target_names=["not_viral", "viral"]
        ),
    }


def save_metrics_txt(metrics: dict, best_params: dict, best_cv_auc: float, path: Path):
    cm = metrics["confusion_matrix"]
    lines = [
        "=" * 62,
        "MODEL B — VIRALITY PREDICTION — EVALUATION REPORT",
        "=" * 62,
        "",
        "BEST HYPERPARAMETERS  (Optuna · 50 trials · 5-fold CV)",
        "-" * 42,
    ]
    for k, v in best_params.items():
        if isinstance(v, float):
            lines.append(f"  {k:<25}  {v:.6g}")
        else:
            lines.append(f"  {k:<25}  {v}")
    lines += [
        "",
        f"  Best CV ROC-AUC (mean) : {best_cv_auc:.4f}",
        "",
        "TEST SET METRICS",
        "-" * 42,
        f"  Accuracy   : {metrics['accuracy']:.4f}",
        f"  Precision  : {metrics['precision']:.4f}  (of predicted viral, how many truly were)",
        f"  Recall     : {metrics['recall']:.4f}  (of all viral, how many were caught)",
        f"  F1 Score   : {metrics['f1']:.4f}",
        f"  ROC-AUC    : {metrics['roc_auc']:.4f}",
        "",
        "CONFUSION MATRIX  (rows = actual, cols = predicted)",
        "-" * 42,
        "                predicted not_viral   predicted viral",
        f"  actual not_viral   {cm[0][0]:<18}  {cm[0][1]}",
        f"  actual viral       {cm[1][0]:<18}  {cm[1][1]}",
        "",
        "CLASSIFICATION REPORT",
        "-" * 42,
        metrics["classification_report"],
        "=" * 62,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_feature_importance(model: LGBMClassifier, path: Path):
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [FEATURE_COLS[i] for i in sorted_idx],
        importances[sorted_idx],
        color="steelblue",
    )
    ax.set_xlabel("Feature Importance (split count)")
    ax.set_title("Model B — LightGBM Feature Importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Feature importance plot  -> {path.name}")


def main():
    log.info("=" * 62)
    log.info("Model B — Virality Prediction Training")
    log.info("=" * 62)
    log.info("")

    ## load + split
    X, y, _ids = load_data()

    n_viral     = int(y.sum())
    n_not_viral = int(len(y) - n_viral)
    spw         = n_not_viral / n_viral
    log.info(f"  viral={n_viral}  not_viral={n_not_viral}  scale_pos_weight={spw:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=SEED
    )
    log.info(f"  train={len(X_train)}  test={len(X_test)}")
    log.info("")

    ## hyperparameter search
    log.info(f"Optuna search  ({N_TRIALS} trials · {N_FOLDS}-fold CV · ROC-AUC) ...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        make_objective(X_train, y_train, spw),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_cv_auc = study.best_value
    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "objective":        "binary",
        "verbosity":        -1,
        "random_state":     SEED,
        "n_jobs":           -1,
    }
    log.info(f"Best CV ROC-AUC : {best_cv_auc:.4f}")
    log.info(f"Best params     : {study.best_params}")
    log.info("")

    ## train final model on full training set
    log.info("Training final model on full training set ...")
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)

    ## evaluate on held-out test set
    log.info("Evaluating on test set ...")
    metrics = evaluate(model, X_test, y_test)

    log.info(f"  Accuracy   : {metrics['accuracy']:.4f}")
    log.info(f"  Precision  : {metrics['precision']:.4f}")
    log.info(f"  Recall     : {metrics['recall']:.4f}")
    log.info(f"  F1         : {metrics['f1']:.4f}")
    log.info(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    log.info("")

    ## save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model           -> {MODEL_PATH.name}")

    ## save metrics.json
    metrics_for_json = {
        k: v for k, v in metrics.items() if k != "classification_report"
    }
    metrics_for_json["best_cv_roc_auc"] = best_cv_auc
    metrics_for_json["best_params"]     = best_params
    metrics_for_json["n_train"]         = len(X_train)
    metrics_for_json["n_test"]          = len(X_test)
    metrics_for_json["feature_names"]   = FEATURE_COLS
    METRICS_JSON.write_text(json.dumps(metrics_for_json, indent=2), encoding="utf-8")
    log.info(f"Metrics JSON    -> {METRICS_JSON.name}")

    ## save metrics.txt (human-readable full report)
    save_metrics_txt(metrics, study.best_params, best_cv_auc, METRICS_TXT)
    log.info(f"Metrics TXT     -> {METRICS_TXT.name}")

    ## feature importance plot
    plot_feature_importance(model, FI_PLOT)

    log.info("")
    log.info("=" * 62)
    log.info("Done.")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
