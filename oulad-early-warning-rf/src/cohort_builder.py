import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import config

def verify_raw_files():
    """Kiểm tra sự tồn tại của các bảng OULAD cần thiết trong data/raw/."""
    required = ["studentInfo.csv", "studentRegistration.csv", "studentVle.csv", "vle.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(config.RAW_DATA_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Thiếu các file dữ liệu gốc trong 'data/raw/': {missing}.\n"
            f"Vui lòng giải nén bộ OULAD (anonymisedData.zip) vào thư mục data/raw/ trước khi chạy!"
        )

def build_cohort():
    """Khảo sát dữ liệu, lọc tuần tự nhóm dự báo tại ngày 28 và xuất cohort_day28.csv."""
    verify_raw_files()
    
    info_path = os.path.join(config.RAW_DATA_DIR, "studentInfo.csv")
    reg_path = os.path.join(config.RAW_DATA_DIR, "studentRegistration.csv")
    
    info = pd.read_csv(info_path)
    reg = pd.read_csv(reg_path)
    
    # Ghép bảng theo bộ ba khóa chính với kiểm tra quan hệ 1-1
    base = info.merge(reg, on=config.PRIMARY_KEY, validate="one_to_one")
    n_init = len(base)
    
    # Bước lọc 1: Nhãn final_result=Withdrawn và date_unregistration phải nhất quán
    cond_consistent = (base["final_result"].eq("Withdrawn") == base["date_unregistration"].notna())
    step1 = base.loc[cond_consistent].copy()
    n_step1 = len(step1)
    
    # Bước lọc 2: Có ngày đăng ký hợp lệ
    cond_has_reg = step1["date_registration"].notna()
    step2 = step1.loc[cond_has_reg].copy()
    n_step2 = len(step2)
    
    # Bước lọc 3: Đăng ký vào hoặc trước ngày 28
    cond_reg_before_cutoff = step2["date_registration"].le(config.CUTOFF_DAY)
    step3 = step2.loc[cond_reg_before_cutoff].copy()
    n_step3 = len(step3)
    
    # Bước lọc 4: Chưa rút học phần tại ngày 28 (date_unregistration là NaN hoặc > 28)
    cond_still_active = step3["date_unregistration"].isna() | step3["date_unregistration"].gt(config.CUTOFF_DAY)
    cohort = step3.loc[cond_still_active].copy()
    n_final = len(cohort)
    
    # Tạo nhãn mục tiêu: Withdrawn vs Not_withdrawn (gộp Fail, Pass, Distinction)
    cohort[config.TARGET_COL] = np.where(
        cohort["final_result"].eq("Withdrawn"), "Withdrawn", "Not_withdrawn"
    )
    
    print("\n" + "="*60)
    print("📊 BẢNG KIỂM TRA LỌC TUẦN TỰ NHÓM DỰ BÁO TẠI NGÀY 28 (OULAD)")
    print("="*60)
    print(f"  • Ban đầu                                : {n_init:,} dòng")
    print(f"  • Loại nhãn & ngày rút không nhất quán   : -{n_init - n_step1:,} -> Còn {n_step1:,} dòng")
    print(f"  • Loại thiếu ngày đăng ký                : -{n_step1 - n_step2:,} -> Còn {n_step2:,} dòng")
    print(f"  • Loại đăng ký sau ngày 28               : -{n_step2 - n_step3:,} -> Còn {n_step3:,} dòng")
    print(f"  • Loại đã rút vào hoặc trước ngày 28     : -{n_step3 - n_final:,} -> Còn {n_final:,} dòng")
    
    n_withdrawn = int((cohort[config.TARGET_COL] == "Withdrawn").sum())
    n_not_withdrawn = int((cohort[config.TARGET_COL] == "Not_withdrawn").sum())
    n_students = cohort["id_student"].nunique()
    print("-"*60)
    print(f"  ✔ Tổng mẫu hợp lệ : {n_final:,} lượt đăng ký ({n_students:,} sinh viên)")
    print(f"  ✔ Not_withdrawn   : {n_not_withdrawn:,} ({n_not_withdrawn/n_final*100:.2f}%)")
    print(f"  ✔ Withdrawn       : {n_withdrawn:,} ({n_withdrawn/n_final*100:.2f}%)")
    print("="*60)
    
    # Lưu file cohort_day28.csv
    out_csv = os.path.join(config.PROCESSED_DATA_DIR, "cohort_day28.csv")
    cohort.to_csv(out_csv, index=False)
    print(f"  💾 Đã lưu nhóm dự báo tại: {out_csv}")
    
    # Vẽ biểu đồ Hình 3.1: Phân phối nhãn
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels = ["Không rút học phần", "Rút học phần"]
    counts = [n_not_withdrawn, n_withdrawn]
    bars = ax.bar(labels, counts, color=["#98b4be", "#3b6978"], width=0.5)
    ax.set_ylabel("Lượt đăng ký")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=4, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    
    fig_path = os.path.join(config.FIGURES_DIR, "fig3_1_label_distribution.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"  📈 Đã lưu biểu đồ phân phối nhãn tại: {fig_path}")
    return cohort