"""Mô hình chính: Random Forest dự báo sinh viên rút học phần (Withdrawn) tại ngày CUTOFF_DAY.
 
Chạy:
    python -m src.main_model            # huấn luyện với RF_PARAMS trong config
    python -m src.main_model --tune     # tinh chỉnh siêu tham số (GroupKFold theo id_student)
"""
import argparse
import json
 
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
 
from src import config
from src.feature_engineering import load_features
from src.predictor import risk_level
 
 
# ------------------------------------------------------------------ dữ liệu
def get_xy(df):
    """X gồm đúng ALL_FEATURES (không có final_result / date_unregistration); y=1 nếu Withdrawn."""
    X = df[config.ALL_FEATURES]
    y = (df[config.TARGET_COL] == config.POSITIVE_LABEL).astype(int)
    return X, y
 
 
def split_data(df):
    """Trả về mảng boolean is_test theo config.SPLIT_MODE."""
    if config.SPLIT_MODE == "temporal":
        return (df["code_presentation"] == config.TEST_PRESENTATION).to_numpy()
    # random: tách theo sinh viên để cùng một id_student không nằm ở cả train và test
    gss = GroupShuffleSplit(n_splits=1, test_size=config.RANDOM_TEST_SIZE, random_state=config.SEED)
    _, te_idx = next(gss.split(df, groups=df["id_student"]))
    mask = np.zeros(len(df), dtype=bool)
    mask[te_idx] = True
    return mask
 
 
# ------------------------------------------------------------------ mô hình
def make_pipeline():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), config.NUMERICAL_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("rf", RandomForestClassifier(**config.RF_PARAMS))])
 
 
def tune(pipe, X, y, groups):
    search = RandomizedSearchCV(
        pipe, config.TUNE_PARAM_GRID, n_iter=config.TUNE_N_ITER,
        cv=GroupKFold(n_splits=config.TUNE_CV), scoring=config.TUNE_SCORING,
        random_state=config.SEED, n_jobs=-1)
    search.fit(X, y, groups=groups)
    print(f"Siêu tham số tốt nhất: {search.best_params_} "
          f"({config.TUNE_SCORING} CV = {search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_
 
 
# ------------------------------------------------------------------ đánh giá
def evaluate(model, X_te, y_te):
    p = model.predict_proba(X_te)[:, 1]
    pred = (p >= config.DECISION_THRESHOLD).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
 
    order = np.argsort(-p)
    k = int(len(p) * 0.2)
    recall_top20 = float(y_te.to_numpy()[order[:k]].sum() / y_te.sum())
 
    return {
        "n_test": int(len(y_te)), "positive_rate_test": round(float(y_te.mean()), 4),
        "roc_auc": round(roc_auc_score(y_te, p), 4),
        "pr_auc": round(average_precision_score(y_te, p), 4),
        "pr_auc_baseline_ngau_nhien": round(float(y_te.mean()), 4),
        "roc_auc_baseline_total_clicks": round(roc_auc_score(y_te, -X_te["total_clicks"]), 4),
        "threshold": config.DECISION_THRESHOLD,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "recall_trong_20pct_rui_ro_cao_nhat": round(recall_top20, 4),
    }, p
 
 
# ------------------------------------------------------------------ Permutation Importance
def plot_permutation_importance(imp, top=config.PERM_TOP_N, path=config.PERM_IMPORTANCE_FIG):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 2))
    ax.barh(d["feature_vi"], d["mean"], xerr=d["std"], color="#2563eb",
            error_kw={"ecolor": "#334155", "capsize": 3})
    pad = d["mean"].max() * 0.01
    for y_pos, (v, sd) in enumerate(zip(d["mean"], d["std"])):
        ax.text(v + sd + pad, y_pos, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, (d["mean"] + d["std"]).max() * 1.15)
    ax.set_xlabel("Mức giảm PR-AUC khi xáo trộn đặc trưng (trên tập test)"
                  if config.PERM_SCORING == "average_precision"
                  else "Mức giảm ROC-AUC khi xáo trộn đặc trưng (trên tập test)")
    ax.set_title(f"Độ quan trọng của đặc trưng (Permutation Importance) - Top {top}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def compute_permutation_importance(model, X_te, y_te):
    r = permutation_importance(model, X_te, y_te, n_repeats=config.PERM_N_REPEATS,
                               scoring=config.PERM_SCORING, random_state=config.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_te.columns, "mean": r.importances_mean, "std": r.importances_std})
    imp["feature_vi"] = imp["feature"].map(config.FEATURE_LABELS_VI)
    imp = imp.sort_values("mean", ascending=False).reset_index(drop=True)
    imp.to_csv(config.PERM_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")
    plot_permutation_importance(imp)
    return imp
 
 
# ------------------------------------------------------------------ quy trình chính
def train_main_model(do_tune=False, verbose=True):
    df = load_features()
    X, y = get_xy(df)
    is_test = split_data(df)
    X_tr, X_te, y_tr, y_te = X[~is_test], X[is_test], y[~is_test], y[is_test]
    if verbose:
        print(f"Chia dữ liệu '{config.SPLIT_MODE}': train {len(X_tr):,} | test {len(X_te):,} | "
              f"Withdrawn train {y_tr.mean():.1%}, test {y_te.mean():.1%}")
 
    pipe = make_pipeline()
    best_params = None
    if do_tune:
        model, best_params = tune(pipe, X_tr, y_tr, df.loc[~is_test, "id_student"])
    else:
        model = pipe.fit(X_tr, y_tr)
 
    metrics, p_te = evaluate(model, X_te, y_te)
    metrics.update({"split_mode": config.SPLIT_MODE, "cutoff_day": config.CUTOFF_DAY,
                    "n_train": int(len(X_tr)), "best_params": best_params})
    imp = compute_permutation_importance(model, X_te, y_te)
 
    joblib.dump({"model": model, "features": config.ALL_FEATURES, "cutoff_day": config.CUTOFF_DAY,
                 "threshold": config.DECISION_THRESHOLD}, config.RF_MODEL_PATH, compress=3)
    with open(config.RF_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
 
    pred = df.loc[is_test, config.PRIMARY_KEY + [config.TARGET_COL]].copy()
    pred["risk_prob"] = p_te.round(4)
    pred["risk_level"] = [risk_level(v) for v in p_te]
    pred.to_csv(config.RF_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
 
    if verbose:
        print("\n=== KẾT QUẢ TRÊN TẬP TEST ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("\n=== PERMUTATION IMPORTANCE (top 10) ===")
        print(imp.head(10)[["feature_vi", "mean", "std"]].to_string(index=False))
        print(f"\nĐã lưu: {config.RF_MODEL_PATH}\n        {config.PERM_IMPORTANCE_FIG}")
    return {"model": model, "metrics": metrics, "importance": imp, "predictions": pred}
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="tinh chỉnh siêu tham số")
    train_main_model(do_tune=ap.parse_args().tune)
 





