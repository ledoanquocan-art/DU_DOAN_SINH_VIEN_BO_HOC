import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import config


def compare_selected_models_menu():
    """
    Mục 4: Đưa ra danh sách các mô hình đã chạy kiểm thử ở Mục 3 (mỗi bộ tham số là 1 mô hình riêng),
    cho phép người dùng chọn từ 2 mô hình trở lên để xuất bảng báo cáo và biểu đồ so sánh.
    """
    metric_files = sorted(glob.glob(os.path.join(config.METRICS_DIR, "*_metrics.json")))
    if len(metric_files) < 2:
        print(f"\n⚠️ Hiện chỉ có {len(metric_files)} mô hình đã được chạy ở [Mục 3].")
        print("   Vui lòng vào [Mục 3. Chạy mô hình] để chạy ít nhất 2 mô hình trước khi Đối chứng!")
        return

    records = []
    for f in metric_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if "model_label" not in data:
                    param_str = ", ".join([f"{k}={v}" for k, v in data.get("params", {}).items()])
                    data["model_label"] = f"{data.get('model_name', 'Model')} [{param_str}]"
                records.append(data)
        except Exception:
            continue

    print("\n" + "=" * 78)
    print("⚖️  [MỤC 4] DANH SÁCH MÔ HÌNH ĐÃ CHẠY (CHỌN >= 2 MÔ HÌNH ĐỂ ĐỐI CHỨNG)")
    print("=" * 78)
    for idx, r in enumerate(records, 1):
        m = r["test_metrics"]
        print(f"  [{idx}] {r['model_label']}\n"
              f"      📊 Acc: {m['accuracy']}% | Macro-F1: {m['macro_f1']}% | "
              f"Precision W: {m['precision_w']}% | Recall W: {m['recall_w']}%")

    raw_sel = input(
        "\n👉 Nhập các số thứ tự mô hình muốn đối chứng, cách nhau bởi dấu phẩy (VD: 1,2,3 hoặc gõ all): "
    ).strip()

    if raw_sel.lower() == "all":
        chosen = records
    else:
        try:
            indices = []
            for x in raw_sel.split(","):
                if x.strip():
                    idx_val = int(x.strip()) - 1
                    if 0 <= idx_val < len(records) and idx_val not in indices:
                        indices.append(idx_val)
            chosen = [records[i] for i in indices]
        except ValueError:
            print("❌ Lựa chọn nhập vào không hợp lệ!")
            return

    if len(chosen) < 2:
        print("❌ Vui lòng chọn từ 2 mô hình trở lên để thực hiện đối chứng!")
        return

    rows = []
    for r in chosen:
        m = r["test_metrics"]
        short_params = ", ".join([f"{k}={v}" for k, v in r.get("params", {}).items()])
        plot_label = f"{r['model_name']}\n({short_params})"
        rows.append({
            "Mô hình & Tham số": r["model_label"],
            "CV Macro-F1 (%)": r.get("cv_macro_f1", "N/A"),
            "Test Accuracy (%)": m["accuracy"],
            "Test Macro-F1 (%)": m["macro_f1"],
            "Precision W (%)": m["precision_w"],
            "Recall W (%)": m["recall_w"],
            "F1 W (%)": m["f1_w"],
            "Đúng W (TP)": m["tp"],
            "Bỏ sót W (FN)": m["fn"],
            "Cảnh báo nhầm (FP)": m["fp"],
            "_plot_label": plot_label
        })

    df_cmp = pd.DataFrame(rows)
    display_cols = [c for c in df_cmp.columns if c != "_plot_label"]

    print("\n" + "=" * 95)
    print("📋 BẢNG BÁO CÁO ĐỐI CHỨNG CÁC MÔ HÌNH ĐÃ CHỌN TRÊN TẬP TEST (5.465 MẪU)")
    print("=" * 95)
    print(df_cmp[display_cols].to_string(index=False))
    print("=" * 95)

    # Lưu bảng báo cáo ra CSV
    csv_path = os.path.join(config.METRICS_DIR, "comparison_report.csv")
    df_cmp[display_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Vẽ biểu đồ cột kép so sánh Accuracy và Macro-F1
    x = np.arange(len(df_cmp))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8.5, len(df_cmp) * 2.4), 5.0))
    b1 = ax.bar(x - width / 2, df_cmp["Test Accuracy (%)"], width, label="Accuracy (%)", color="#3b6978")
    b2 = ax.bar(x + width / 2, df_cmp["Test Macro-F1 (%)"], width, label="Macro-F1 (%)", color="#98b4be")

    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x, labels=df_cmp["_plot_label"], fontsize=9)
    ax.set_title("Đối chứng Accuracy và Macro-F1 trên Tập Test giữa các mô hình đã chọn")
    ax.legend()
    ax.bar_label(b1, padding=3, fmt="%.1f", fontsize=9)
    ax.bar_label(b2, padding=3, fmt="%.1f", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig_path = os.path.join(config.FIGURES_DIR, "fig3_3_model_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"  💾 Đã lưu bảng báo cáo đối chứng (CSV) : {csv_path}")
    print(f"  📊 Đã lưu biểu đồ đối chứng (PNG)      : {fig_path}")