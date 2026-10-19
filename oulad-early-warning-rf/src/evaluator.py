import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
from src import config
from src.pipeline_utils import list_saved_models, load_and_split_data

def plot_confusion_matrix(tn, fp, fn, tp, model_label, save_path):
    """Vẽ và lưu Ma trận nhầm lẫn chuẩn báo cáo."""
    mat = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Không rút", "Rút học phần"])
    ax.set_yticks([0, 1], labels=["Không rút", "Rút học phần"])
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

def run_saved_model_menu():
    """Mục 3: Chạy mô hình trên tập Test chung và xuất biểu đồ."""
    models = list_saved_models()
    if not models:
        print("\n⚠️ Chưa có mô hình nào được huấn luyện! Vui lòng vào [Mục 2] để huấn luyện trước.")
        return

    print("\n" + "=" * 75)
    print("🚀 [MỤC 3] CHỌN MÔ HÌNH ĐÃ HUẤN LUYỆN ĐỂ CHẠY KIỂM THỬ & XUẤT BIỂU ĐỒ")
    print("=" * 75)
    for idx, m in enumerate(models, 1):
        status = f"✅ Đã chạy Test (Macro-F1: {m['test_metrics']['macro_f1']}%)" if m.get("test_metrics") else "🆕 Chưa chạy Test"
        print(f"  [{idx}] {m['model_label']} | CV Macro-F1: {m['cv_macro_f1']}% -> {status}")

    choice = input(f"\n👉 Nhập số thứ tự mô hình muốn chạy (1-{len(models)}, hoặc 0 để quay lại): ").strip()
    if not choice.isdigit() or int(choice) == 0 or int(choice) > len(models):
        return

    selected = models[int(choice) - 1]
    pipe = selected["pipeline"]
    model_id = selected["model_id"]
    model_label = selected["model_label"]

    # Gọi hàm cắt dữ liệu chuẩn để lấy X_test, y_test
    _, _, _, X_test, y_test, _, _ = load_and_split_data()
    print(f"\n⏳ Đang chạy mô hình [{model_label}] trên {len(X_test):,} mẫu Test chung...")
    y_pred = pipe.predict(X_test)

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

    # 1. Vẽ Ma trận nhầm lẫn
    cm_fig_path = os.path.join(config.FIGURES_DIR, f"cm_{model_id}.png")
    plot_confusion_matrix(tn, fp, fn, tp, model_label, cm_fig_path)

    # 2. Permutation Importance (Chỉ vẽ cho Random Forest)
    pi_path = None
    if selected["model_key"] == "rf":
        print("⏳ Đang tính Permutation Importance (5 lần lặp) cho cấu hình Random Forest này...")
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

    # 3. Lưu cập nhật
    selected["test_metrics"] = metrics
    filepath = selected.pop("filepath")
    joblib.dump(selected, filepath)

    json_payload = {
        "model_id": model_id,
        "model_key": selected["model_key"],
        "model_name": selected["model_name"],
        "model_label": model_label,
        "params": selected["params"],
        "cv_accuracy": selected["cv_accuracy"],
        "cv_macro_f1": selected["cv_macro_f1"],
        "test_metrics": metrics
    }
    json_path = os.path.join(config.METRICS_DIR, f"{model_id}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"📊 KẾT QUẢ KIỂM THỬ: {model_label}")
    print("=" * 65)
    print(f"  • Accuracy          : {metrics['accuracy']}%")
    print(f"  • Macro-F1          : {metrics['macro_f1']}%")
    print(f"  • Precision (Rút)   : {metrics['precision_w']}%")
    print(f"  • Recall (Rút)      : {metrics['recall_w']}%")
    print(f"  • Chi tiết lớp Rút  : Đúng (TP) = {tp} | Bỏ sót (FN) = {fn} | Cảnh báo nhầm (FP) = {fp}")
    print(f"  • Đã lưu biểu đồ CM : results/figures/cm_{model_id}.png")