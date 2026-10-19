"""Dự đoán nguy cơ rút học phần cho file không nhãn.

Hàm gọi từ main.py (menu 5): predict_unlabeled_file_menu()
"""
import os
import subprocess
import sys

import joblib
import pandas as pd

from src import config


def predict_unlabeled_file_menu():
    """Nhập file CSV không nhãn -> dự đoán bằng Random Forest -> lưu kết quả và mở file."""
    model_path = os.path.join(config.MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(model_path):
        print("\n⚠️ Chưa có mô hình Random Forest! Vui lòng chạy Mục 2 -> 1 (Huấn luyện) trước.")
        return
    model = joblib.load(model_path)

    print("\n" + "="*65)
    print("🔮 DỰ ĐOÁN NGUY CƠ RÚT HỌC PHẦN TỪ FILE KHÔNG NHÃN")
    print("="*65)
    print(f"File CSV cần có đủ {len(config.ALL_FEATURES)} cột đặc trưng (cột target nếu có sẽ bị bỏ qua).")
    path = input("👉 Nhập đường dẫn file CSV (Enter = lấy mẫu 20 sinh viên khóa 2014J): ").strip().strip('"')

    if path == "":
        feat_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
        if not os.path.exists(feat_path):
            print("⚠️ Chưa có features_day28.csv! Vui lòng chạy Mục 1 (Tiền xử lý) trước.")
            return
        df = pd.read_csv(feat_path)
        df = df[df["code_presentation"] == "2014J"].sample(20, random_state=config.SEED)
        cot_giu = list(dict.fromkeys(config.PRIMARY_KEY + config.ALL_FEATURES))   # code_module có ở cả hai danh sách
        df = df[cot_giu]                                              # bỏ nhãn -> thành file không nhãn
        path = os.path.join(config.PREDICTIONS_DIR, "mau_khong_nhan.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"   Đã tạo file mẫu: {path}")
    elif not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
        return
    else:
        df = pd.read_csv(path)

    thieu = [c for c in config.ALL_FEATURES if c not in df.columns]
    if thieu:
        print(f"❌ File thiếu {len(thieu)} cột đặc trưng, ví dụ: {thieu[:5]}")
        return

    # Dự đoán: xác suất Withdrawn, mức rủi ro và nhãn dự đoán
    xac_suat = model.predict_proba(df[config.ALL_FEATURES])[:, 1]
    ket_qua = df.drop(columns=[config.TARGET_COL], errors="ignore").copy()
    ket_qua["risk_prob"] = xac_suat.round(4)
    ket_qua["risk_level"] = pd.cut(xac_suat, [0, 0.30, 0.50, 1.0001], right=False, include_lowest=True,
                                   labels=["Thấp", "Trung bình", "Cao"])
    ket_qua["predicted_label"] = ["Withdrawn" if p >= 0.5 else "Not_withdrawn" for p in xac_suat]

    out_path = os.path.join(config.PREDICTIONS_DIR, os.path.splitext(os.path.basename(path))[0] + "_du_doan.csv")
    ket_qua.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Đã dự đoán {len(ket_qua):,} sinh viên. Phân bố mức rủi ro:")
    print(ket_qua["risk_level"].value_counts().reindex(["Thấp", "Trung bình", "Cao"], fill_value=0).to_string())
    cot = [c for c in config.PRIMARY_KEY if c in ket_qua.columns] + ["risk_prob", "risk_level", "predicted_label"]
    print("\nTop 5 sinh viên có nguy cơ cao nhất:")
    print(ket_qua.sort_values("risk_prob", ascending=False)[cot].head(5).to_string(index=False))
    print(f"\n💾 Đã lưu kết quả: {out_path}")

    try:  # mở file kết quả
        if sys.platform.startswith("win"):
            os.startfile(out_path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", out_path])
        print("📂 Đã mở file kết quả.")
    except Exception:
        print("ℹ️ Không tự mở được file, bạn hãy mở thủ công ở đường dẫn trên.")
