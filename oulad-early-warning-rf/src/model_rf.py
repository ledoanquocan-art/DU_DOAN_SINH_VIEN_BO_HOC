"""Random Forest: huấn luyện, đánh giá, Permutation Importance và lưu mô hình.

Hàm gọi từ main.py (menu 2 -> 1): train_rf()
  Đầu vào : data/processed/features_day28.csv (do generate_features() tạo)
  Đầu ra  : results/models/random_forest.joblib
            results/metrics/random_forest_metrics.json
            results/figures/fig_permutation_importance.png
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config

TEST_PRESENTATION = "2014J"   # test = khóa mới nhất, train = các khóa cũ hơn
THRESHOLD = 0.5               # xác suất >= ngưỡng này thì dự đoán là Withdrawn

# Tên tiếng Việt cho biểu đồ
TEN_VI = {
    "code_module": "Mã môn học", "gender": "Giới tính", "region": "Khu vực",
    "highest_education": "Trình độ học vấn", "imd_band": "Nhóm thu nhập khu vực (IMD)",
    "age_band": "Nhóm tuổi", "disability": "Khuyết tật",
    "num_of_prev_attempts": "Số lần học lại", "studied_credits": "Số tín chỉ đăng ký",
    "days_since_registration": "Số ngày kể từ khi đăng ký", "total_clicks": "Tổng số lượt click",
    "active_days": "Số ngày có hoạt động", "unique_sites": "Số trang VLE đã truy cập",
    "recency_days": "Số ngày kể từ lần hoạt động gần nhất",
    "mean_clicks_active_day": "Click trung bình mỗi ngày hoạt động",
    "clicks_d0_6": "Click ngày 0-6", "clicks_d7_13": "Click ngày 7-13",
    "clicks_d14_20": "Click ngày 14-20", "clicks_d21_28": "Click ngày 21-28",
}
TEN_VI_HOAT_DONG = {
    "dataplus": "dữ liệu bổ sung", "dualpane": "giao diện hai khung", "externalquiz": "bài kiểm tra ngoài",
    "folder": "thư mục", "forumng": "diễn đàn", "glossary": "bảng thuật ngữ", "homepage": "trang chủ",
    "htmlactivity": "hoạt động HTML", "oucollaborate": "phòng cộng tác", "oucontent": "nội dung bài học",
    "ouelluminate": "lớp học trực tuyến", "ouwiki": "wiki", "page": "trang nội dung",
    "questionnaire": "bảng khảo sát", "quiz": "bài kiểm tra", "repeatactivity": "hoạt động lặp lại",
    "resource": "tài liệu", "sharedsubpage": "trang con chia sẻ", "subpage": "trang con", "url": "liên kết ngoài",
}
for _act, _vi in TEN_VI_HOAT_DONG.items():
    TEN_VI[f"activity_clicks_{_act}"] = f"Click: {_vi}"


def _nhap_so(ten, mac_dinh):
    """Nhập một số nguyên; bỏ trống hoặc nhập sai thì dùng giá trị mặc định."""
    s = input(f"   {ten} [mặc định {mac_dinh}]: ").strip()
    return int(s) if s.isdigit() else mac_dinh


def train_rf():
    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
    if not os.path.exists(feat_path):
        print("\n⚠️ Chưa có features_day28.csv! Vui lòng chạy Mục 1 (Tiền xử lý) trước.")
        return
    df = pd.read_csv(feat_path)

    # --- 1. Nhập tham số ---
    print("\nNhập tham số Random Forest (Enter = giữ mặc định):")
    n_estimators = _nhap_so("Số cây (n_estimators)", 300)
    max_depth = _nhap_so("Độ sâu tối đa (max_depth)", 16)
    min_samples_leaf = _nhap_so("Số mẫu tối thiểu mỗi lá (min_samples_leaf)", 5)

    # --- 2. Chia train/test theo thời gian: test = khóa 2014J ---
    X = df[config.ALL_FEATURES]
    y = (df[config.TARGET_COL] == "Withdrawn").astype(int)      # 1 = Withdrawn (lớp cần phát hiện)
    la_test = df["code_presentation"] == TEST_PRESENTATION
    X_train, y_train, X_test, y_test = X[~la_test], y[~la_test], X[la_test], y[la_test]
    print(f"\nTrain: {len(X_train):,} dòng | Test (khóa {TEST_PRESENTATION}): {len(X_test):,} dòng "
          f"| Withdrawn: {y_train.mean():.1%} / {y_test.mean():.1%}")

    # --- 3. Huấn luyện ---
    print("⏳ Đang huấn luyện Random Forest...")
    tien_xu_ly = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), config.NUMERICAL_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), config.CATEGORICAL_FEATURES),
    ])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=config.SEED)
    model = Pipeline([("pre", tien_xu_ly), ("rf", rf)]).fit(X_train, y_train)

    # --- 4. Đánh giá trên tập test ---
    xac_suat = model.predict_proba(X_test)[:, 1]
    du_doan = (xac_suat >= THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, du_doan, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, du_doan).ravel()
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, xac_suat), 4),
        "pr_auc": round(average_precision_score(y_test, xac_suat), 4),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "tham_so": {"n_estimators": n_estimators, "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf},
    }
    print("\n=== KẾT QUẢ TRÊN TẬP TEST ===")
    print(f"  ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} "
          f"(ngẫu nhiên = {y_test.mean():.3f})")
    print(f"  Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1']}")
    print(f"  Ma trận nhầm lẫn: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # --- 5. Permutation Importance (trên tập test) ---
    print("\n⏳ Đang tính Permutation Importance...")
    kq = permutation_importance(model, X_test, y_test, n_repeats=10, scoring="roc_auc",
                                random_state=config.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_test.columns, "ten_vi": [TEN_VI[c] for c in X_test.columns],
                        "mean": kq.importances_mean, "std": kq.importances_std})
    imp = imp.sort_values("mean", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(config.METRICS_DIR, "permutation_importance.csv"),
               index=False, encoding="utf-8-sig")
    print(imp.head(10)[["ten_vi", "mean", "std"]].to_string(index=False))

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["ten_vi"], top["mean"], xerr=top["std"], color="#2563eb",
            error_kw={"ecolor": "#334155", "capsize": 3})
    for i, (m, s) in enumerate(zip(top["mean"], top["std"])):
        ax.text(m + s + top["mean"].max() * 0.01, i, f"{m:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, (top["mean"] + top["std"]).max() * 1.15)
    ax.set_xlabel("Mức giảm ROC-AUC khi xáo trộn đặc trưng (trên tập test)")
    ax.set_title("Độ quan trọng của đặc trưng (Permutation Importance) - Top 15")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(config.FIGURES_DIR, "fig_permutation_importance.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # --- 6. Lưu mô hình và số liệu ---
    model_path = os.path.join(config.MODELS_DIR, "random_forest.joblib")
    joblib.dump(model, model_path, compress=3)
    metrics_path = os.path.join(config.METRICS_DIR, "random_forest_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Đã lưu mô hình : {model_path}")
    print(f"📈 Đã lưu biểu đồ : {fig_path}")
    print(f"📄 Đã lưu số liệu : {metrics_path}")
