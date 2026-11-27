import os
import json
import sys
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src import config


# Các cột lõi có ảnh hưởng lớn đến độ chính xác dự đoán
CORE_REQUIRED_FEATURES = [
    "code_module",
    "clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28",
    "active_days", "recency_days"
]

# Các cột dẫn xuất có thể tự động tính lại từ 4 cột click theo tuần & active_days
DERIVED_WEEKLY_FEATURES = [
    "total_clicks",
    "mean_clicks_active_day",
    "click_trend_w4_vs_w1",
    "click_trend_w4_vs_w3",
    "zero_click_weeks",
    "trailing_zero_weeks",
    "click_std_4w"
]


def _extract_model_and_meta(loaded_obj, filename):
    """Tách pipeline mô hình và metadata từ file .joblib (hỗ trợ cả dạng dict lẫn Pipeline trực tiếp)."""
    if isinstance(loaded_obj, dict):
        model = loaded_obj.get("pipeline") or loaded_obj.get("model") or loaded_obj.get("estimator")
        meta = {
            "model_name": loaded_obj.get("model_name", filename),
            "model_key": loaded_obj.get("model_key", ""),
            "params": loaded_obj.get("params", {}),
            "cv_macro_f1": loaded_obj.get("cv_macro_f1"),
            "cv_accuracy": loaded_obj.get("cv_accuracy"),
        }
        return model, meta

    return loaded_obj, {
        "model_name": filename,
        "model_key": "",
        "params": {},
        "cv_macro_f1": None,
        "cv_accuracy": None,
    }


def _load_metrics_for_model(model_stem):
    """Tìm và đọc file chỉ số kiểm thử JSON tương ứng trong results/metrics/ (nếu đã chạy Mục 3)."""
    candidates = [
        os.path.join(config.METRICS_DIR, f"{model_stem}_metrics.json"),
        os.path.join(config.METRICS_DIR, f"{model_stem}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def list_saved_models():
    """
    Quét toàn bộ file .joblib trong results/models/ và ghép thêm chỉ số đánh giá
    (Test Accuracy / Macro-F1 / Recall hoặc CV Macro-F1) để người dùng dễ chọn mô hình.
    """
    if not os.path.exists(config.MODELS_DIR):
        return []

    records = []
    for fname in sorted(os.listdir(config.MODELS_DIR)):
        if not fname.endswith(".joblib"):
            continue
        full_path = os.path.join(config.MODELS_DIR, fname)
        stem = os.path.splitext(fname)[0]
        metrics = _load_metrics_for_model(stem)

        # Đọc nhanh metadata từ artifact nếu chưa có file metrics JSON
        cv_f1 = None
        display_name = stem
        try:
            obj = joblib.load(full_path)
            _, meta = _extract_model_and_meta(obj, fname)
            display_name = meta.get("model_name", stem)
            if meta.get("params"):
                param_str = ", ".join(f"{k}={v}" for k, v in meta["params"].items())
                display_name = f"{display_name} [{param_str}]"
            cv_f1 = meta.get("cv_macro_f1")
        except Exception:
            pass

        test_acc = metrics.get("accuracy")
        test_f1 = metrics.get("macro_f1")
        test_recall = metrics.get("recall_withdrawn")

        # Tạo nhãn tóm tắt độ chính xác để hiển thị trong Menu / Selectbox
        if test_acc is not None and test_f1 is not None:
            acc_pct = test_acc * 100 if test_acc <= 1.0 else test_acc
            f1_pct = test_f1 * 100 if test_f1 <= 1.0 else test_f1
            score_tag = f"Test Acc: {acc_pct:.2f}% | Macro-F1: {f1_pct:.2f}%"
            if test_recall is not None:
                rec_pct = test_recall * 100 if test_recall <= 1.0 else test_recall
                score_tag += f" | Recall(W): {rec_pct:.2f}%"
        elif cv_f1 is not None:
            cv_pct = cv_f1 * 100 if cv_f1 <= 1.0 else cv_f1
            score_tag = f"Chưa chạy Test | CV Macro-F1: {cv_pct:.2f}%"
        else:
            score_tag = "Chưa có chỉ số kiểm thử"

        records.append({
            "filename": fname,
            "stem": stem,
            "path": full_path,
            "display_name": display_name,
            "score_tag": score_tag,
            "label": f"{display_name}  ({score_tag})",
            "metrics": metrics,
        })

    return records


def get_model_expected_features(model):
    """Tự động lấy đúng danh sách cột mà mô hình đã được huấn luyện (39 hoặc 50 đặc trưng)."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return list(config.ALL_FEATURES)


def prepare_unlabeled_features(df_raw, model):
    """
    Chuẩn hóa dữ liệu đầu vào cho tính năng số 5 (hỗ trợ cả file đầy đủ lẫn file rút gọn):
      1. Xác định chính xác tập cột mà mô hình yêu cầu (39 hoặc 50 cột).
      2. Chuẩn hóa chuỗi phân loại (strip khoảng trắng, giữ nguyên NaN).
      3. Tự động tính lại các cột dẫn xuất (total_clicks, mean_clicks_active_day, 5 cột trend)
         nếu file CSV đã có 4 cột click tuần.
      4. Điền 0 cho các cột activity_clicks_* bị thiếu (đúng bản chất log VLE OULAD)
         và điền NaN cho các cột không quan trọng còn lại để SimpleImputer trong Pipeline xử lý.

    Trả về:
      - X_ready: DataFrame đúng thứ tự và đủ số cột đầu vào của mô hình.
      - report: dict thống kê (expected_count, provided_count, auto_computed, missing_core,
                filled_zero_vle, filled_nan_other).
    """
    expected_cols = get_model_expected_features(model)
    df = df_raw.copy()

    # Loại khoảng trắng thừa ở tên cột (nếu người dùng tạo CSV từ Excel)
    df.columns = [str(c).strip() for c in df.columns]

    # Chuẩn hóa ký hiệu khuyết ẩn ('?', '') nếu DataFrame chưa được đọc với na_values
    df.replace(["?", ""], np.nan, inplace=True)

    # Làm sạch khoảng trắng thừa ở các cột phân loại hiện có
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            mask_not_na = df[col].notna()
            df.loc[mask_not_na, col] = df.loc[mask_not_na, col].astype(str).str.strip()

    provided_expected = [c for c in expected_cols if c in df.columns]
    auto_computed = []

    # 1. Tự tính lại các cột dẫn xuất từ 4 cột click theo tuần (nếu thiếu)
    weekly_cols = ["clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28"]
    if all(c in df.columns for c in weekly_cols):
        w1 = pd.to_numeric(df["clicks_d0_6"], errors="coerce").fillna(0.0)
        w2 = pd.to_numeric(df["clicks_d7_13"], errors="coerce").fillna(0.0)
        w3 = pd.to_numeric(df["clicks_d14_20"], errors="coerce").fillna(0.0)
        w4 = pd.to_numeric(df["clicks_d21_28"], errors="coerce").fillna(0.0)
        weekly_mat = np.column_stack([w1, w2, w3, w4])

        if "total_clicks" in expected_cols and "total_clicks" not in df.columns:
            df["total_clicks"] = (w1 + w2 + w3 + w4).astype(int)
            auto_computed.append("total_clicks")

        if "click_trend_w4_vs_w1" in expected_cols and "click_trend_w4_vs_w1" not in df.columns:
            df["click_trend_w4_vs_w1"] = ((w4 + 1.0) / (w1 + 1.0)).round(4)
            auto_computed.append("click_trend_w4_vs_w1")

        if "click_trend_w4_vs_w3" in expected_cols and "click_trend_w4_vs_w3" not in df.columns:
            df["click_trend_w4_vs_w3"] = ((w4 + 1.0) / (w3 + 1.0)).round(4)
            auto_computed.append("click_trend_w4_vs_w3")

        if "zero_click_weeks" in expected_cols and "zero_click_weeks" not in df.columns:
            df["zero_click_weeks"] = (weekly_mat == 0).sum(axis=1).astype(int)
            auto_computed.append("zero_click_weeks")

        if "trailing_zero_weeks" in expected_cols and "trailing_zero_weeks" not in df.columns:
            df["trailing_zero_weeks"] = np.select(
                [w4 > 0, w3 > 0, w2 > 0, w1 > 0],
                [0, 1, 2, 3],
                default=4
            ).astype(int)
            auto_computed.append("trailing_zero_weeks")

        if "click_std_4w" in expected_cols and "click_std_4w" not in df.columns:
            df["click_std_4w"] = np.std(weekly_mat, axis=1).round(4)
            auto_computed.append("click_std_4w")

    if (
        "mean_clicks_active_day" in expected_cols
        and "mean_clicks_active_day" not in df.columns
        and "total_clicks" in df.columns
        and "active_days" in df.columns
    ):
        act = pd.to_numeric(df["active_days"], errors="coerce").fillna(0.0)
        tot = pd.to_numeric(df["total_clicks"], errors="coerce").fillna(0.0)
        df["mean_clicks_active_day"] = np.where(act > 0, (tot / act).round(4), 0.0)
        auto_computed.append("mean_clicks_active_day")

    # 2. Kiểm tra cột lõi bị thiếu & tự bổ sung cột ít quan trọng còn lại
    missing_core = [c for c in CORE_REQUIRED_FEATURES if c in expected_cols and c not in df.columns]
    filled_zero_vle = []
    filled_nan_other = []

    for col in expected_cols:
        if col not in df.columns:
            if col.startswith("activity_clicks_"):
                df[col] = 0
                filled_zero_vle.append(col)
            else:
                df[col] = np.nan
                filled_nan_other.append(col)

    # Ép kiểu số an toàn cho toàn bộ biến số trong expected_cols
    num_set = set(config.NUMERICAL_FEATURES)
    for col in expected_cols:
        if col in num_set:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X_ready = df[expected_cols].copy()
    report = {
        "expected_count": len(expected_cols),
        "provided_count": len(provided_expected),
        "auto_computed": auto_computed,
        "missing_core": missing_core,
        "filled_zero_vle": filled_zero_vle,
        "filled_nan_other": filled_nan_other,
    }
    return X_ready, report


def _classify_risk_tier(prob_pct, pred_label):
    """Phân tầng mức độ nguy cơ rút học phần phục vụ cố vấn học tập."""
    if prob_pct is None or pd.isna(prob_pct):
        return "Cần ưu tiên liên hệ" if pred_label == "Withdrawn" else "Bình thường"
    if prob_pct >= 65.0:
        return "Rất cao"
    if prob_pct >= 50.0:
        return "Cao"
    if prob_pct >= 35.0:
        return "Trung bình"
    return "Thấp"


def predict_unlabeled_dataframe(df_raw, model_path, batch_size=2000, progress_callback=None):
    """
    Hàm lõi dự đoán dữ liệu không nhãn từ DataFrame (dùng chung cho cả CLI và Streamlit).

    Trả về:
      - df_result: DataFrame kết quả gồm cột định danh, nhãn dự đoán, xác suất & mức nguy cơ + cột gốc.
      - summary: dict tổng hợp kết quả & báo cáo tương thích cột.
    """
    if progress_callback:
        progress_callback(10, "Đang nạp mô hình đã huấn luyện...")

    loaded = joblib.load(model_path)
    model, meta = _extract_model_and_meta(loaded, os.path.basename(model_path))
    if model is None:
        raise ValueError(f"Không tìm thấy đối tượng Pipeline/Estimator hợp lệ trong {model_path}")

    if progress_callback:
        progress_callback(25, "Đang chuẩn hóa và đồng bộ đặc trưng đầu vào...")

    X_ready, align_report = prepare_unlabeled_features(df_raw, model)
    n_rows = len(X_ready)
    if n_rows == 0:
        raise ValueError("File dữ liệu đầu vào không có bản ghi nào (0 dòng).")

    # Dự đoán theo lô để cập nhật thanh tiến trình mượt mà trên tập dữ liệu lớn
    preds_all = []
    probs_w_all = []
    has_proba = False

    # Kiểm tra mô hình có hỗ trợ predict_proba hay không
    try:
        sample_proba = model.predict_proba(X_ready.iloc[:1])
        classes = list(model.classes_)
        if "Withdrawn" in classes and sample_proba is not None:
            has_proba = True
            w_idx = classes.index("Withdrawn")
    except Exception:
        has_proba = False
        w_idx = None

    n_batches = max(1, int(np.ceil(n_rows / batch_size)))
    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_rows)
        X_batch = X_ready.iloc[start:end]

        preds_all.append(model.predict(X_batch))
        if has_proba:
            p_batch = model.predict_proba(X_batch)[:, w_idx] * 100.0
            probs_w_all.append(np.round(p_batch, 2))

        if progress_callback:
            pct = 30 + int(((b + 1) / n_batches) * 60)
            progress_callback(pct, f"Đang dự đoán dữ liệu ({end:,}/{n_rows:,} dòng)...")

    y_pred = np.concatenate(preds_all)
    probs_w = np.concatenate(probs_w_all) if has_proba else np.full(n_rows, np.nan)

    # Xây dựng bảng đầu ra rõ ràng (đặt cột định danh & kết quả dự đoán lên đầu)
    df_out = df_raw.copy()
    # Tránh nhầm lẫn nếu file test người dùng vô tình vẫn còn cột nhãn cũ
    for old_col in ["predicted_label", "predicted_status_vi", "withdrawn_probability_pct", "risk_level"]:
        if old_col in df_out.columns:
            df_out.drop(columns=[old_col], inplace=True)

    status_vi = np.where(y_pred == "Withdrawn", "Rút học phần", "Không rút học phần")
    risk_levels = [_classify_risk_tier(p, lbl) for p, lbl in zip(probs_w, y_pred)]

    # Tách cột định danh nếu có để đưa lên đầu bảng kết quả
    id_cols = [c for c in ["id_student", "code_module", "code_presentation"] if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in id_cols]

    result_front = pd.DataFrame(index=df_out.index)
    for c in id_cols:
        result_front[c] = df_out[c]

    result_front["predicted_label"] = y_pred
    result_front["predicted_status_vi"] = status_vi
    result_front["withdrawn_probability_pct"] = probs_w if has_proba else "N/A"
    result_front["risk_level"] = risk_levels

    df_result = pd.concat([result_front, df_out[other_cols]], axis=1)

    n_withdrawn = int((y_pred == "Withdrawn").sum())
    n_not_withdrawn = int(n_rows - n_withdrawn)

    if progress_callback:
        progress_callback(100, "Hoàn tất dự đoán dữ liệu không nhãn!")

    summary = {
        "model_name": meta.get("model_name", os.path.basename(model_path)),
        "model_file": os.path.basename(model_path),
        "total_rows": n_rows,
        "withdrawn_count": n_withdrawn,
        "not_withdrawn_count": n_not_withdrawn,
        "withdrawn_rate_pct": round(n_withdrawn / n_rows * 100.0, 2),
        "has_proba": has_proba,
        "align_report": align_report,
    }
    return df_result, summary


def predict_unlabeled_csv(input_csv_path, model_path, output_csv_path=None, progress_callback=None):
    """Đọc file CSV không nhãn, chạy dự đoán và lưu file kết quả vào data/predictions/."""
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Không tìm thấy tệp CSV đầu vào: {input_csv_path}")

    df_raw = pd.read_csv(input_csv_path, na_values=["?", ""])
    df_result, summary = predict_unlabeled_dataframe(
        df_raw=df_raw,
        model_path=model_path,
        progress_callback=progress_callback
    )

    if output_csv_path is None:
        os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
        in_stem = os.path.splitext(os.path.basename(input_csv_path))[0]
        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(
            config.PREDICTIONS_DIR,
            f"pred_{in_stem}_{model_stem}_{timestamp}.csv"
        )

    df_result.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    summary["output_path"] = output_csv_path
    return df_result, summary


def _open_file_os(filepath):
    """Mở trực tiếp file kết quả bằng ứng dụng mặc định của hệ điều hành."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(filepath)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            subprocess.Popen(["xdg-open", filepath])
        return True
    except Exception:
        return False


def predict_unlabeled_menu():
    """
    Mục 5 trên giao diện dòng lệnh (CLI - main.py):
    Chọn mô hình đã lưu (kèm độ chính xác), nạp file CSV không nhãn (đầy đủ hoặc rút gọn),
    dự đoán, lưu file và hỗ trợ mở trực tiếp.
    """
    print("\n" + "=" * 78)
    print("🔮 [MỤC 5] DỰ ĐOÁN NGUY CƠ RÚT HỌC PHẦN TRÊN DỮ LIỆU KHÔNG NHÃN (.CSV)")
    print("=" * 78)

    models = list_saved_models()
    if not models:
        print("⚠️  Chưa có mô hình nào được huấn luyện trong results/models/.")
        print("👉 Vui lòng chạy [Mục 2] để huấn luyện ít nhất một mô hình trước.")
        return

    print("📦 Danh sách mô hình sẵn có:")
    for idx, m in enumerate(models, 1):
        print(f"   [{idx}] {m['display_name']}")
        print(f"       └─ {m['score_tag']}  |  File: {m['filename']}")

    choice = input("\n👉 Chọn số thứ tự mô hình muốn dùng (hoặc 0 để quay lại): ").strip()
    if not choice.isdigit() or int(choice) == 0:
        return
    idx_sel = int(choice) - 1
    if not (0 <= idx_sel < len(models)):
        print("❌ Lựa chọn không hợp lệ.")
        return

    selected_model = models[idx_sel]
    print(f"✔ Đã chọn mô hình: {selected_model['display_name']}")

    print("\n💡 Lưu ý: File CSV có thể ở dạng đầy đủ (39/50 cột) HOẶC dạng rút gọn chỉ gồm các cột quan trọng.")
    raw_path = input("👉 Nhập đường dẫn tới file .csv không nhãn (kéo thả file vào terminal hoặc gõ đường dẫn): ").strip()
    csv_path = raw_path.strip('"').strip("'")

    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy tệp tại đường dẫn: {csv_path}")
        return

    def cli_progress(pct, msg):
        print(f"   [{pct:3d}%] {msg}")

    try:
        df_res, summary = predict_unlabeled_csv(
            input_csv_path=csv_path,
            model_path=selected_model["path"],
            progress_callback=cli_progress
        )
    except Exception as exc:
        print(f"❌ Lỗi trong quá trình dự đoán: {exc}")
        return

    rep = summary["align_report"]
    print("\n" + "-" * 78)
    print("📋 BÁO CÁO ĐỒNG BỘ ĐẶC TRƯNG ĐẦU VÀO")
    print("-" * 78)
    print(f"  • Số đặc trưng mô hình yêu cầu : {rep['expected_count']} cột")
    print(f"  • Số cột khớp trực tiếp từ CSV : {rep['provided_count']} cột")
    if rep["auto_computed"]:
        print(f"  • Tự tính lại từ 4 cột tuần    : {len(rep['auto_computed'])} cột {rep['auto_computed']}")
    if rep["filled_zero_vle"]:
        print(f"  • Cột VLE thưa tự điền 0       : {len(rep['filled_zero_vle'])} cột")
    if rep["filled_nan_other"]:
        print(f"  • Cột phụ tự điền NaN (Imputer): {len(rep['filled_nan_other'])} cột {rep['filled_nan_other']}")
    if rep["missing_core"]:
        print(f"  🚨 CẢNH BÁO: File thiếu các cột lõi {rep['missing_core']} -> độ tin cậy dự đoán sẽ giảm!")

    print("\n" + "-" * 78)
    print("📊 KẾT QUẢ DỰ ĐOÁN TỔNG HỢP")
    print("-" * 78)
    print(f"  • Tổng số hồ sơ dự đoán        : {summary['total_rows']:,} dòng")
    print(f"  • Cảnh báo Rút học phần        : {summary['withdrawn_count']:,} ca ({summary['withdrawn_rate_pct']:.2f}%)")
    print(f"  • Không rút học phần           : {summary['not_withdrawn_count']:,} ca ({100 - summary['withdrawn_rate_pct']:.2f}%)")
    print(f"  • Đã lưu file kết quả tại      : {summary['output_path']}")

    # Xem trước 10 dòng đầu
    preview_cols = [
        c for c in ["id_student", "code_module", "predicted_label", "predicted_status_vi",
                    "withdrawn_probability_pct", "risk_level"]
        if c in df_res.columns
    ]
    print("\n👀 Xem trước kết quả (tối đa 10 dòng đầu):")
    print(df_res[preview_cols].head(10).to_string(index=False))

    open_ans = input("\n👉 Bạn có muốn mở trực tiếp file kết quả ngay bây giờ không? (y/n): ").strip().lower()
    if open_ans in ("y", "yes", "c", "co"):
        if _open_file_os(summary["output_path"]):
            print("✔ Đã gửi lệnh mở tệp kết quả.")
        else:
            print(f"⚠️  Không thể mở tự động, vui lòng mở thủ công tại: {summary['output_path']}")


# Alias giữ tương thích nếu main.py gọi theo tên cũ
run_predictor = predict_unlabeled_menu
predict_new_data = predict_unlabeled_menu