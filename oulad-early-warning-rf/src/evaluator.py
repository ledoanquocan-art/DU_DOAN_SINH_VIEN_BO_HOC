import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
from src import config
from src.pipeline_utils import list_saved_models, load_and_split_data, print_progress_bar


def plot_confusion_matrix(tn, fp, fn, tp, model_label, save_path):
    """Vẽ và lưu Ma trận nhầm lẫn chuẩn báo cáo."""
    mat = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Không bỏ học", "Bỏ học"])
    ax.set_yticks([0, 1], labels=["Không bỏ học", "Bỏ học"])
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(f"Ma trận nhầm lẫn\n{model_label}", fontsize=10.5)

    for i in range(2):
        for j in range(2):
            color = "white" if mat[i][j] > (tn + fp + fn + tp) / 4 else "black"
            ax.text(j, i, f"{mat[i][j]:,}", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def evaluate_model_with_progress(selected, progress_callback=None):
    """Thực thi kiểm thử mô hình trên tập Test với thanh % tiến trình chi tiết."""
    def update_prog(pct, msg):
        print_progress_bar(pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

    pipe = selected["pipeline"]
    model_id = selected["model_id"]
    model_label = selected["model_label"]

    print(f"\n🚀 Đang chạy kiểm thử mô hình: {model_label}")
    update_prog(10, "Đang nạp tập dữ liệu Test (test_day28.csv)...")
    _, X_train, _, X_test, y_test, _, _ = load_and_split_data(save_split_files=True)

    # Chia tập Test thành 5 batch để cập nhật thanh tiến trình % từ 15% -> 65%
    n_samples = len(X_test)
    n_batches = 5
    batch_size = int(np.ceil(n_samples / n_batches))
    preds = []

    for b in range(n_batches):
        start_i = b * batch_size
        end_i = min((b + 1) * batch_size, n_samples)
        if start_i >= end_i:
            break
        batch_X = X_test.iloc[start_i:end_i]
        preds.append(pipe.predict(batch_X))
        pct = 15 + int(((b + 1) / n_batches) * 50)
        update_prog(pct, f"Đang dự đoán trên tập Test ({end_i:,}/{n_samples:,} mẫu)...")

    y_pred = pd.Series(np.concatenate(preds), index=X_test.index)

    update_prog(72, "Đang tính toán các chỉ số Accuracy, Macro-F1, Recall...")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["Not_withdrawn", "Withdrawn"]).ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")) * 100, 2),
        "precision_w": round(float(precision_score(y_test, y_pred, pos_label="Withdrawn", zero_division=0)) * 100, 2),
        "recall_w": round(float(recall_score(y_test, y_pred, pos_label="Withdrawn", zero_division=0)) * 100, 2),
        "f1_w": round(float(f1_score(y_test, y_pred, pos_label="Withdrawn", zero_division=0)) * 100, 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        classes = list(pipe.named_steps["model"].classes_)
        w_idx = classes.index("Withdrawn")
        y_prob = pipe.predict_proba(X_test)[:, w_idx]
        y_true_bin = (y_test == "Withdrawn").astype(int)
        metrics["ap_withdrawn"] = round(float(average_precision_score(y_true_bin, y_prob)), 4)
        metrics["roc_auc"] = round(float(roc_auc_score(y_true_bin, y_prob)), 4)

    update_prog(80, "Đang vẽ và lưu biểu đồ Ma trận nhầm lẫn...")
    cm_fig_path = os.path.join(config.FIGURES_DIR, f"cm_{model_id}.png")
    plot_confusion_matrix(tn, fp, fn, tp, model_label, cm_fig_path)

    pi_path = None
    if selected["model_key"] == "rf":
        update_prog(88, "Đang tính Permutation Importance (5 lần lặp)...")
        perm = permutation_importance(
            pipe, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=config.SEED, n_jobs=2
        )
        sorted_idx = perm.importances_mean.argsort()[-8:]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(
            [config.ALL_FEATURES[i] for i in sorted_idx],
            perm.importances_mean[sorted_idx] * 100,
            xerr=perm.importances_std[sorted_idx] * 100,
            color="#3b6978"
        )
        ax.set_xlabel("Mức giảm Macro-F1 khi hoán vị (điểm phần trăm)")
        ax.set_title(f"Top 8 Đặc trưng quan trọng nhất\n{model_label}", fontsize=10)
        fig.tight_layout()
        pi_path = os.path.join(config.FIGURES_DIR, f"perm_importance_{model_id}.png")
        plt.savefig(pi_path, dpi=200)
        plt.close(fig)

    update_prog(96, "Đang lưu kết quả JSON và cập nhật mô hình...")
    selected_copy = dict(selected)
    filepath = selected_copy.pop("filepath", os.path.join(config.MODELS_DIR, f"{model_id}.joblib"))
    selected_copy["test_metrics"] = metrics
    joblib.dump(selected_copy, filepath)

    json_payload = {
        "model_id": model_id,
        "model_key": selected["model_key"],
        "model_name": selected["model_name"],
        "model_label": model_label,
        "params": selected["params"],
        "cv_accuracy": selected["cv_accuracy"],
        "cv_macro_f1": selected["cv_macro_f1"],
        "train_data_path": config.TRAIN_DATA_PATH,
        "test_data_path": config.TEST_DATA_PATH,
        "test_metrics": metrics
    }
    json_path = os.path.join(config.METRICS_DIR, f"{model_id}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=4, ensure_ascii=False)

    update_prog(100, "Hoàn tất 100% kiểm thử mô hình và xuất biểu đồ!")

    print("\n" + "=" * 78)
    print(f"📊 KẾT QUẢ KIỂM THỬ MÔ HÌNH: {model_label}")
    print("-" * 78)
    print(f"  • Tập dữ liệu Train     : {config.TRAIN_DATA_PATH} ({len(X_train):,} mẫu)")
    print(f"  • Tập dữ liệu Test      : {config.TEST_DATA_PATH} ({len(X_test):,} mẫu)")
    print(f"  • Accuracy              : {metrics['accuracy']}%")
    print(f"  • Macro-F1              : {metrics['macro_f1']}%")
    print(f"  • Precision (bỏ học)       : {metrics['precision_w']}%")
    print(f"  • Recall (bỏ học)          : {metrics['recall_w']}%")
    print(f"  • F1-Score (bỏ học)        : {metrics['f1_w']}%")
    print(f"  • Chi tiết lớp bỏ học      : Đúng (TP) = {tp} | Bỏ sót (FN) = {fn} | Cảnh báo nhầm (FP) = {fp}")
    print(f"  • File kết quả JSON     : {json_path}")
    print(f"  • Biểu đồ CM đã lưu     : {cm_fig_path}")
    if pi_path:
        print(f"  • Biểu đồ Đặc trưng     : {pi_path}")
    print("=" * 78)
    return metrics, cm_fig_path, pi_path


def run_saved_model_menu():
    """Mục 3: Chọn mô hình đã lưu để chạy trên tập Test chung có thanh % tiến trình."""
    models = list_saved_models()
    if not models:
        print("\n⚠️ Chưa có mô hình nào được huấn luyện! Vui lòng vào [Mục 2] để huấn luyện trước.")
        return

    print("\n" + "=" * 78)
    print("🚀 [MỤC 3] CHỌN MÔ HÌNH ĐÃ HUẤN LUYỆN ĐỂ CHẠY KIỂM THỬ & XUẤT BIỂU ĐỒ")
    print("=" * 78)
    for idx, m in enumerate(models, 1):
        status = f"✅ Đã chạy Test (Macro-F1: {m['test_metrics']['macro_f1']}%)" if m.get("test_metrics") else "🆕 Chưa chạy Test"
        print(f"  [{idx}] {m['model_label']} | CV Macro-F1: {m['cv_macro_f1']}% -> {status}")

    choice = input(f"\n👉 Nhập số thứ tự mô hình muốn chạy (1-{len(models)}, hoặc 0 để quay lại): ").strip()
    if not choice.isdigit() or int(choice) == 0 or int(choice) > len(models):
        return

    selected = models[int(choice) - 1]
    evaluate_model_with_progress(selected)