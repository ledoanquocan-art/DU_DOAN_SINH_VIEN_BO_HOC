import numpy as np
import os
import pandas as pd
from src import config
from src.pipeline_utils import list_saved_models, load_and_split_data, open_file_on_os

# Nhóm cột lõi tối thiểu cần có để đảm bảo độ tin cậy dự đoán
CORE_REQUIRED_FEATURES = [
    "code_module",
    "clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28",
    "active_days", "recency_days"
]


def prepare_unlabeled_features(df_raw, model):
def create_sample_unlabeled_file():
    """
    Chuẩn hóa file CSV rút gọn đầu vào cho tính năng số 5:
      1. Tự tính lại các cột dẫn xuất (total_clicks, mean_clicks_active_day, 5 cột trend) nếu thiếu.
      2. Điền 0 cho các cột activity_clicks_* bị thiếu (đúng bản chất log VLE).
      3. Điền NaN cho các cột ít quan trọng còn lại để SimpleImputer trong Pipeline tự xử lý.
    Tạo sẵn 1 file CSV không nhãn mẫu (gồm dòng demo 334 + 49 dòng từ tập Test)
    để người dùng có thể nhấn Enter chạy thử ngay ở Mục 5.
    """
    expected_cols = (
        list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else list(config.ALL_FEATURES)
    df_full, _, _, _, _, _, test_idx = load_and_split_data()

    # Đưa dòng 334 lên đầu file mẫu để tiện đối chiếu báo cáo, kèm 49 mẫu trong tập Test
    selected_indices = [334] + [i for i in test_idx[:50] if i != 334][:49]
    sample_df = df_full.iloc[selected_indices].drop(
        columns=[config.TARGET_COL, "final_result"], errors="ignore"
    )

    df = df_raw.copy()
    weekly_cols = ["clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28"]
    has_weekly = all(c in df.columns for c in weekly_cols)

    # 1. Tự tính lại các cột dẫn xuất từ 4 tuần click (tránh mất tín hiệu nếu file CSV rút gọn bỏ qua)
    if has_weekly:
        w1 = pd.to_numeric(df["clicks_d0_6"], errors="coerce").fillna(0.0)
        w2 = pd.to_numeric(df["clicks_d7_13"], errors="coerce").fillna(0.0)
        w3 = pd.to_numeric(df["clicks_d14_20"], errors="coerce").fillna(0.0)
        w4 = pd.to_numeric(df["clicks_d21_28"], errors="coerce").fillna(0.0)
        weekly_mat = np.column_stack([w1, w2, w3, w4])

        if "total_clicks" in expected_cols and "total_clicks" not in df.columns:
            df["total_clicks"] = (w1 + w2 + w3 + w4).astype(int)
        if "click_trend_w4_vs_w1" in expected_cols and "click_trend_w4_vs_w1" not in df.columns:
            df["click_trend_w4_vs_w1"] = ((w4 + 1.0) / (w1 + 1.0)).round(4)
        if "click_trend_w4_vs_w3" in expected_cols and "click_trend_w4_vs_w3" not in df.columns:
            df["click_trend_w4_vs_w3"] = ((w4 + 1.0) / (w3 + 1.0)).round(4)
        if "zero_click_weeks" in expected_cols and "zero_click_weeks" not in df.columns:
            df["zero_click_weeks"] = (weekly_mat == 0).sum(axis=1).astype(int)
        if "trailing_zero_weeks" in expected_cols and "trailing_zero_weeks" not in df.columns:
            df["trailing_zero_weeks"] = np.select([w4 > 0, w3 > 0, w2 > 0, w1 > 0], [0, 1, 2, 3], default=4).astype(int)
        if "click_std_4w" in expected_cols and "click_std_4w" not in df.columns:
            df["click_std_4w"] = np.std(weekly_mat, axis=1).round(4)

    if (
        "mean_clicks_active_day" in expected_cols
        and "mean_clicks_active_day" not in df.columns
        and "total_clicks" in df.columns
        and "active_days" in df.columns
    ):
        act = pd.to_numeric(df["active_days"], errors="coerce").fillna(0.0)
        tot = pd.to_numeric(df["total_clicks"], errors="coerce").fillna(0.0)
        df["mean_clicks_active_day"] = np.where(act > 0, (tot / act).round(4), 0.0)

    # 2. Kiểm tra cột lõi và điền giá trị mặc định cho các cột không quan trọng bị thiếu
    missing_core = [c for c in CORE_REQUIRED_FEATURES if c in expected_cols and c not in df.columns]
    missing_optional = []

    for col in expected_cols:
        if col not in df.columns:
            missing_optional.append(col)
            if col.startswith("activity_clicks_"):
                df[col] = 0
            else:
                df[col] = np.nan

    X_ready = df[expected_cols]
    return X_ready, missing_core, missing_optional
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    sample_path = os.path.join(config.PREDICTIONS_DIR, "sample_unlabeled_students.csv")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    return sample_path


def predict_unlabeled_file_menu():
    """
    Mục 5: Nhập file dữ liệu không nhãn -> Chọn mô hình (kèm tham số & độ chính xác)
    -> Trả về file dữ liệu đã gán nhãn và tự động mở file trên máy.
    """
    models = list_saved_models()
    if not models:
        print("\n⚠️ Chưa có mô hình nào được huấn luyện! Vui lòng vào [Mục 2] để huấn luyện trước.")
        return

    print("\n" + "=" * 75)
    print("🔮 [MỤC 5] DỰ ĐOÁN DỮ LIỆU KHÔNG NHÃN")
    print("=" * 75)

    default_sample = create_sample_unlabeled_file()
    print(f"💡 Đã chuẩn bị sẵn file mẫu không nhãn tại: {default_sample}")
    file_path = input("👉 Nhập đường dẫn file CSV không nhãn (Nhấn Enter để dùng file mẫu trên): ").strip()
    if not file_path:
        file_path = default_sample

    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file tại đường dẫn: {file_path}")
        return

    df_input = pd.read_csv(file_path)
    missing_cols = [c for c in config.ALL_FEATURES if c not in df_input.columns]
    if missing_cols:
        print(f"❌ File đầu vào thiếu các cột đặc trưng bắt buộc: {missing_cols}")
        return

    print("\n" + "-" * 75)
    print("📋 DANH SÁCH MÔ HÌNH ĐÃ HUẤN LUYỆN (KÈM THAM SỐ & ĐỘ CHÍNH XÁC):")
    print("-" * 75)
    for idx, m in enumerate(models, 1):
        label = m.get("model_label", f"{m['model_name']} [{m['params']}]")
        if m.get("test_metrics"):
            tm = m["test_metrics"]
            acc_info = f"Test Accuracy: {tm['accuracy']}% | Test Macro-F1: {tm['macro_f1']}% | Recall W: {tm['recall_w']}%"
        else:
            acc_info = f"CV Accuracy: {m['cv_accuracy']}% | CV Macro-F1: {m['cv_macro_f1']}% (Chưa chạy Test)"
        print(f"  [{idx}] {label}\n      🎯 Độ chính xác -> {acc_info}")

    choice = input(f"\n👉 Chọn số thứ tự mô hình muốn dùng để dự đoán (1-{len(models)}, hoặc 0 để hủy): ").strip()
    if not choice.isdigit() or int(choice) == 0 or int(choice) > len(models):
        return

    chosen = models[int(choice) - 1]
    pipe = chosen["pipeline"]
    model_id = chosen["model_id"]
    model_label = chosen.get("model_label", chosen["model_name"])

    X_unlabeled = df_input[config.ALL_FEATURES]
    df_out = df_input.copy()
    df_out["predicted_target"] = pipe.predict(X_unlabeled)

    # Tính xác suất rút học phần nếu mô hình hỗ trợ predict_proba (như RF, k-NN, NB)
    try:
        classes = list(pipe.named_steps["model"].classes_)
        w_idx = classes.index("Withdrawn")
        probs = pipe.predict_proba(X_unlabeled)[:, w_idx] * 100
        df_out["withdrawn_probability_%"] = probs.round(2)
    except Exception:
        # Trường hợp SVM để probability=False
        df_out["withdrawn_probability_%"] = "N/A (SVM probability=False)"

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_filename = f"{base_name}__predicted_by__{model_id}.csv"
    out_path = os.path.join(config.PREDICTIONS_DIR, out_filename)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_total = len(df_out)
    n_withdrawn = int((df_out["predicted_target"] == "Withdrawn").sum())
    n_not_withdrawn = n_total - n_withdrawn

    print("\n" + "=" * 75)
    print(f"✅ DỰ ĐOÁN HOÀN TẤT BẰNG MÔ HÌNH: {model_label}")
    print("=" * 75)
    print(f"  • Tổng số bản ghi dự đoán           : {n_total:,} dòng")
    print(f"  • Dự đoán bỏ học (Withdrawn)  : {n_withdrawn:,} ca ({n_withdrawn/n_total*100:.2f}%)")
    print(f"  • Dự đoán Không bỏ học (Not_withdrawn) : {n_not_withdrawn:,} ca ({n_not_withdrawn/n_total*100:.2f}%)")
    if len(df_out) > 0:
        first_row = df_out.iloc[0]
        print(f"  • Minh họa bản ghi đầu tiên         : SV={first_row.get('id_student', 'N/A')} | "
              f"Học phần={first_row.get('code_module', 'N/A')} -> Nhãn={first_row['predicted_target']} "
              f"(Xác suất W: {first_row['withdrawn_probability_%']}%)")
    print(f"  • Đã lưu file có nhãn tại           : data/predictions/{out_filename}")
    print("=" * 75)

    # Mở trực tiếp file kết quả trên màn hình
    open_file_on_os(out_path)