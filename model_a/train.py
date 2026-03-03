## Model A — XGBoost Cascade Training (Step 2 of 2)

## Loads the per-video feature vectors produced by extract_features.py
## and trains two XGBoost binary classifiers arranged as a cascade:

##   Submodel 2  AI-Generated Detector
##               Input:  first 2048 features (EfficientNet + ForensicCNN only)
##               Target: 1 = ai_generated,  0 = real or deepfake
##               Trained on all three classes combined.

##   Submodel 1  Real vs Deepfake Classifier
##               Input:  all 2064 features (EfficientNet + ForensicCNN + MediaPipe)
##               Target: 1 = deepfake,  0 = real
##               Trained only on real and deepfake rows (ai_generated excluded
##               because those videos never reach Submodel 1 in the cascade).

## Cascade logic at inference:
##   video -> Submodel 2 -> prob >= AI_THRESH  -> label = AI_GENERATED
##                       -> else               -> Submodel 1
##                            -> prob >= DF_THRESH -> label = DEEPFAKE
##                            -> else              -> label = REAL

## Outputs saved to model_a/:
##   submodel1.json    — Real vs Deepfake XGBoost model
##   submodel2.json    — AI-Generated XGBoost model
##   metrics.json      — all evaluation numbers as a JSON dict
##   metrics.txt       — human-readable full report

## Run with:  uv run model_a/train.py

import json
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT         = Path(__file__).parent.parent
OUT_DIR      = ROOT / "model_a"
TRAIN_NPZ    = OUT_DIR / "features_train.npz"
TEST_NPZ     = OUT_DIR / "features_test.npz"
SM1_PATH     = OUT_DIR / "submodel1.json"
SM2_PATH     = OUT_DIR / "submodel2.json"
METRICS_JSON = OUT_DIR / "metrics.json"
METRICS_TXT  = OUT_DIR / "metrics.txt"

## small JSON files saved after each submodel finishes so a crashed run can resume
SM1_CKPT = OUT_DIR / ".sm1_checkpoint.json"
SM2_CKPT = OUT_DIR / ".sm2_checkpoint.json"

## pause this many seconds every PAUSE_EVERY_TRIALS Optuna trials
PAUSE_EVERY_TRIALS = 10
PAUSE_SECS         = 10

## label encoding — must match extract_features.py LABEL_MAP
REAL         = 0
DEEPFAKE     = 1
AI_GENERATED = 2

## cascade decision thresholds (tunable post-training)
AI_THRESH = 0.50
DF_THRESH = 0.50

## Optuna / CV settings
SEED     = 42
N_TRIALS = 50
N_FOLDS  = 5


def load_split(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return (
        d["X"].astype(np.float32),
        d["y"].astype(int),
        d["video_ids"],
        d["sources"],
    )


def log_source_breakdown(y: np.ndarray, sources: np.ndarray, split: str):
    """Log how many real / deepfake / ai_generated videos came from each source."""
    label_names = {REAL: "real", DEEPFAKE: "deepfake", AI_GENERATED: "ai_gen"}
    from collections import defaultdict
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, src in zip(y, sources):
        counts[str(src)][label_names[int(label)]] += 1
    log.info(f"  {split} breakdown by source:")
    for src in sorted(counts):
        c = counts[src]
        parts = "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
        log.info(f"    {src:<35} {parts}")
    log.info("")


def make_xgb_objective(X_train: np.ndarray, y_train: np.ndarray, spw: float):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "scale_pos_weight": spw,
            "objective":        "binary:logistic",
            "verbosity":        0,
            "random_state":     SEED,
            "n_jobs":           -1,
        }
        scores = cross_val_score(
            xgb.XGBClassifier(**params),
            X_train, y_train,
            cv=cv,
            scoring="roc_auc",
        )
        return float(scores.mean())

    return objective


def optuna_pause_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    ## called after every trial — pause briefly every PAUSE_EVERY_TRIALS to let the CPU cool down
    if (trial.number + 1) % PAUSE_EVERY_TRIALS == 0:
        log.info(f"  [pause] {PAUSE_SECS}s after trial {trial.number + 1} ...")
        time.sleep(PAUSE_SECS)


def tune_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    spw: float,
    name: str,
) -> tuple[xgb.XGBClassifier, float, dict]:
    log.info(f"  Optuna: {N_TRIALS} trials, {N_FOLDS}-fold CV, ROC-AUC ...")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        make_xgb_objective(X_train, y_train, spw),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        callbacks=[optuna_pause_callback],
    )

    best_params = {
        **study.best_params,
        "scale_pos_weight": spw,
        "objective":        "binary:logistic",
        "verbosity":        0,
        "random_state":     SEED,
        "n_jobs":           -1,
    }
    log.info(f"  Best CV ROC-AUC: {study.best_value:.4f}  params: {study.best_params}")

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_value, best_params


def evaluate_binary(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50,
    target_names: list[str] | None = None,
) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        "accuracy":              float(accuracy_score(y_test, y_pred)),
        "precision":             float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":                float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                    float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":               float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix":      confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names or ["negative", "positive"]
        ),
    }


def cascade_predict(
    sub2: xgb.XGBClassifier,
    sub1: xgb.XGBClassifier,
    X: np.ndarray,
) -> np.ndarray:
    ## Submodel 2 uses only EfficientNet + ForensicCNN features (first 2048 dims)
    prob_ai = sub2.predict_proba(X[:, :2048])[:, 1]
    ## Submodel 1 uses all 2064 features
    prob_df = sub1.predict_proba(X)[:, 1]

    return np.where(
        prob_ai >= AI_THRESH, AI_GENERATED,
        np.where(prob_df >= DF_THRESH, DEEPFAKE, REAL),
    )


def evaluate_cascade(
    sub2: xgb.XGBClassifier,
    sub1: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    preds = cascade_predict(sub2, sub1, X_test)

    f1_per = f1_score(
        y_test, preds,
        average=None,
        labels=[REAL, DEEPFAKE, AI_GENERATED],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=[REAL, DEEPFAKE, AI_GENERATED])

    return {
        "cascade_accuracy":              float(accuracy_score(y_test, preds)),
        "cascade_f1_real":               float(f1_per[0]),
        "cascade_f1_deepfake":           float(f1_per[1]),
        "cascade_f1_ai_generated":       float(f1_per[2]),
        "cascade_macro_f1":              float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "cascade_confusion_matrix":      cm.tolist(),
        "cascade_classification_report": classification_report(
            y_test, preds, target_names=["real", "deepfake", "ai_generated"]
        ),
    }


def save_metrics_txt(
    sm1_m: dict, sm1_cv: float,
    sm2_m: dict, sm2_cv: float,
    cascade_m: dict,
    path: Path,
):
    cm1 = sm1_m["confusion_matrix"]
    cm2 = sm2_m["confusion_matrix"]
    cc  = cascade_m["cascade_confusion_matrix"]

    lines = [
        "=" * 62,
        "MODEL A — SYNTHETIC MEDIA DETECTION — EVALUATION REPORT",
        "=" * 62,
        "",
        "SUBMODEL 2 — AI-Generated Detector  (binary: ai_gen vs rest)",
        "-" * 42,
        f"  Best CV ROC-AUC : {sm2_cv:.4f}",
        f"  Test Accuracy   : {sm2_m['accuracy']:.4f}",
        f"  Test Precision  : {sm2_m['precision']:.4f}",
        f"  Test Recall     : {sm2_m['recall']:.4f}",
        f"  Test F1         : {sm2_m['f1']:.4f}",
        f"  Test ROC-AUC    : {sm2_m['roc_auc']:.4f}",
        "",
        "  Confusion Matrix  (rows=actual, cols=predicted)",
        "                       not_ai    ai_gen",
        f"  actual not_ai        {cm2[0][0]:<10}  {cm2[0][1]}",
        f"  actual ai_gen        {cm2[1][0]:<10}  {cm2[1][1]}",
        "",
        sm2_m["classification_report"],
        "",
        "SUBMODEL 1 — Real vs Deepfake  (binary: deepfake vs real)",
        "-" * 42,
        f"  Best CV ROC-AUC : {sm1_cv:.4f}",
        f"  Test Accuracy   : {sm1_m['accuracy']:.4f}",
        f"  Test Precision  : {sm1_m['precision']:.4f}",
        f"  Test Recall     : {sm1_m['recall']:.4f}",
        f"  Test F1         : {sm1_m['f1']:.4f}",
        f"  Test ROC-AUC    : {sm1_m['roc_auc']:.4f}",
        "",
        "  Confusion Matrix  (rows=actual, cols=predicted)",
        "                       real      deepfake",
        f"  actual real          {cm1[0][0]:<10}  {cm1[0][1]}",
        f"  actual deepfake      {cm1[1][0]:<10}  {cm1[1][1]}",
        "",
        sm1_m["classification_report"],
        "",
        "CASCADE — Full 3-Class Evaluation on Test Set",
        "-" * 42,
        f"  Accuracy         : {cascade_m['cascade_accuracy']:.4f}",
        f"  Macro F1         : {cascade_m['cascade_macro_f1']:.4f}",
        f"  F1 (real)        : {cascade_m['cascade_f1_real']:.4f}",
        f"  F1 (deepfake)    : {cascade_m['cascade_f1_deepfake']:.4f}",
        f"  F1 (ai_gen)      : {cascade_m['cascade_f1_ai_generated']:.4f}",
        "",
        "  Confusion Matrix  (rows=actual, cols=predicted: real / deepfake / ai_gen)",
        f"  actual real        {cc[0][0]:<10}  {cc[0][1]:<10}  {cc[0][2]}",
        f"  actual deepfake    {cc[1][0]:<10}  {cc[1][1]:<10}  {cc[1][2]}",
        f"  actual ai_gen      {cc[2][0]:<10}  {cc[2][1]:<10}  {cc[2][2]}",
        "",
        cascade_m["cascade_classification_report"],
        "=" * 62,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    log.info("=" * 62)
    log.info("Model A — XGBoost Cascade Training")
    log.info("=" * 62)
    log.info("")

    X_train, y_train, _, src_train = load_split(TRAIN_NPZ)
    X_test,  y_test,  _, src_test  = load_split(TEST_NPZ)

    log.info(f"Feature shape — train: {X_train.shape}  test: {X_test.shape}")
    log.info(
        f"Train labels — real: {(y_train==REAL).sum()}  "
        f"deepfake: {(y_train==DEEPFAKE).sum()}  "
        f"ai_gen: {(y_train==AI_GENERATED).sum()}"
    )
    log.info(
        f"Test  labels — real: {(y_test==REAL).sum()}  "
        f"deepfake: {(y_test==DEEPFAKE).sum()}  "
        f"ai_gen: {(y_test==AI_GENERATED).sum()}"
    )
    log.info("")
    log_source_breakdown(y_train, src_train, "train")
    log_source_breakdown(y_test,  src_test,  "test")

    ## Submodel 2 is the cascade entry point — it catches AI-generated videos first

    ## only Stream 1 + Stream 2 features for this submodel (no MediaPipe)
    X_tr2 = X_train[:, :2048]
    X_te2 = X_test[:, :2048]
    y_tr2 = (y_train == AI_GENERATED).astype(int)
    y_te2 = (y_test  == AI_GENERATED).astype(int)

    if SM2_PATH.exists() and SM2_CKPT.exists():
        log.info("Submodel 2 already trained — loading from disk, skipping retraining ...")
        sub2 = xgb.XGBClassifier()
        sub2.load_model(str(SM2_PATH))
        ckpt2    = json.loads(SM2_CKPT.read_text())
        sm2_cv   = ckpt2["best_cv"]
        sm2_params = ckpt2["best_params"]
    else:
        log.info("Training Submodel 2 — AI-Generated Detector ...")
        n_neg2 = int((y_tr2 == 0).sum())
        n_pos2 = int((y_tr2 == 1).sum())
        spw2   = n_neg2 / max(n_pos2, 1)
        log.info(f"  not_ai={n_neg2}  ai_gen={n_pos2}  scale_pos_weight={spw2:.2f}")
        sub2, sm2_cv, sm2_params = tune_and_train(X_tr2, y_tr2, spw2, "Submodel2")
        sub2.save_model(str(SM2_PATH))
        SM2_CKPT.write_text(json.dumps({"best_cv": sm2_cv, "best_params": sm2_params}))
        log.info(f"  Saved -> {SM2_PATH.name}")

    sm2_m = evaluate_binary(sub2, X_te2, y_te2, AI_THRESH, ["not_ai", "ai_generated"])
    log.info(
        f"  Accuracy={sm2_m['accuracy']:.4f}  Precision={sm2_m['precision']:.4f}  "
        f"Recall={sm2_m['recall']:.4f}  F1={sm2_m['f1']:.4f}  ROC-AUC={sm2_m['roc_auc']:.4f}"
    )
    log.info("")

    log.info(f"[pause] 10s before starting Submodel 1 ...")
    time.sleep(10)

    ## Submodel 1 only sees real and deepfake videos — ai_generated never reaches here

    ## keep only real and deepfake rows
    mask_tr = (y_train == REAL) | (y_train == DEEPFAKE)
    mask_te = (y_test  == REAL) | (y_test  == DEEPFAKE)

    X_tr1 = X_train[mask_tr]
    y_tr1 = (y_train[mask_tr] == DEEPFAKE).astype(int)
    X_te1 = X_test[mask_te]
    y_te1 = (y_test[mask_te]  == DEEPFAKE).astype(int)

    if SM1_PATH.exists() and SM1_CKPT.exists():
        log.info("Submodel 1 already trained — loading from disk, skipping retraining ...")
        sub1 = xgb.XGBClassifier()
        sub1.load_model(str(SM1_PATH))
        ckpt1    = json.loads(SM1_CKPT.read_text())
        sm1_cv   = ckpt1["best_cv"]
        sm1_params = ckpt1["best_params"]
    else:
        log.info("Training Submodel 1 — Real vs Deepfake ...")
        n_real = int((y_tr1 == 0).sum())
        n_fake = int((y_tr1 == 1).sum())
        spw1   = n_real / max(n_fake, 1)
        log.info(f"  real={n_real}  deepfake={n_fake}  scale_pos_weight={spw1:.2f}")
        sub1, sm1_cv, sm1_params = tune_and_train(X_tr1, y_tr1, spw1, "Submodel1")
        sub1.save_model(str(SM1_PATH))
        SM1_CKPT.write_text(json.dumps({"best_cv": sm1_cv, "best_params": sm1_params}))
        log.info(f"  Saved -> {SM1_PATH.name}")

    sm1_m = evaluate_binary(sub1, X_te1, y_te1, DF_THRESH, ["real", "deepfake"])
    log.info(
        f"  Accuracy={sm1_m['accuracy']:.4f}  Precision={sm1_m['precision']:.4f}  "
        f"Recall={sm1_m['recall']:.4f}  F1={sm1_m['f1']:.4f}  ROC-AUC={sm1_m['roc_auc']:.4f}"
    )
    log.info("")

    ## run the full cascade on the test set to get the real-world 3-class numbers
    log.info("Evaluating full cascade on test set ...")
    cascade_m = evaluate_cascade(sub2, sub1, X_test, y_test)
    log.info(
        f"  Cascade Accuracy={cascade_m['cascade_accuracy']:.4f}  "
        f"Macro F1={cascade_m['cascade_macro_f1']:.4f}"
    )
    log.info(
        f"  F1 — real={cascade_m['cascade_f1_real']:.4f}  "
        f"deepfake={cascade_m['cascade_f1_deepfake']:.4f}  "
        f"ai_gen={cascade_m['cascade_f1_ai_generated']:.4f}"
    )
    log.info("")

    ## save both models and all metrics
    metrics_out = {
        "submodel2": {
            **{k: v for k, v in sm2_m.items() if k != "classification_report"},
            "best_cv_roc_auc": sm2_cv,
            "best_params":     sm2_params,
            "n_train":         int(X_tr2.shape[0]),
            "n_test":          int(X_te2.shape[0]),
            "feature_dims":    2048,
        },
        "submodel1": {
            **{k: v for k, v in sm1_m.items() if k != "classification_report"},
            "best_cv_roc_auc": sm1_cv,
            "best_params":     sm1_params,
            "n_train":         int(X_tr1.shape[0]),
            "n_test":          int(X_te1.shape[0]),
            "feature_dims":    2064,
        },
        "cascade": {
            k: v for k, v in cascade_m.items()
            if k != "cascade_classification_report"
        },
        "thresholds": {
            "ai_threshold": AI_THRESH,
            "df_threshold": DF_THRESH,
        },
    }

    METRICS_JSON.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log.info(f"Metrics JSON -> {METRICS_JSON.name}")

    save_metrics_txt(sm1_m, sm1_cv, sm2_m, sm2_cv, cascade_m, METRICS_TXT)
    log.info(f"Metrics TXT  -> {METRICS_TXT.name}")

    log.info("")
    log.info("=" * 62)
    log.info("Done.")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
