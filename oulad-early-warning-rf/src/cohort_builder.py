import os
import pandas as pd
import matplotlib.pyplot as plt
from src import config


REQUIRED_RAW_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
    "vle.csv",
    "studentVle.csv",
    "assessments.csv",
    "studentAssessment.csv"
]


def verify_raw_files():
    """Kiểm tra đầy đủ 7 bảng gốc của bộ dữ liệu OULAD trong data/raw/."""
    missing = [
        f for f in REQUIRED_RAW_FILES
        if not os.path.exists(os.path.join(config.RAW_DATA_DIR, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"Thiếu các file OULAD gốc trong {config.RAW_DATA_DIR}: {missing}"
        )
    print("✔ Đã xác nhận đủ 7 bảng dữ liệu gốc OULAD trong data/raw/.")


def plot_label_distribution(df, save_path):
    """Vẽ và lưu biểu đồ phân phối nhãn (Hình 3.1)."""
    counts = df[config.TARGET_COL].value_counts().reindex(["Not_withdrawn", "Withdrawn"])
    total = len(df)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(
        ["Không rút\n(Not_withdrawn)", "Rút học phần\n(Withdrawn)"],
        counts.values,
        color=["#3b6978", "#d97706"],
        width=0.52
    )
    ax.set_ylabel("Số lượt đăng ký")
    ax.set_title(f"Phân phối nhãn trên {total:,} lượt đăng ký tại ngày {config.CUTOFF_DAY}")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + total * 0.015,
            f"{val:,}\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def build_cohort():
    """
    Khảo sát dữ liệu gốc, lọc 4 bước chống rò rỉ dữ liệu tại ngày 28 (t = 28),
    tạo nhãn nhị phân target và xuất cohort_day28.csv.
    """
    verify_raw_files()

    info_path = os.path.join(config.RAW_DATA_DIR, "studentInfo.csv")
    reg_path = os.path.join(config.RAW_DATA_DIR, "studentRegistration.csv")

    # Đọc và chuẩn hóa các ký hiệu missing ('?', '') về NaN
    df_info = pd.read_csv(info_path, na_values=["?", ""])
    df_reg = pd.read_csv(reg_path, na_values=["?", ""])

    # Ép kiểu số an toàn cho ngày đăng ký và ngày rút môn
    df_reg["date_registration"] = pd.to_numeric(df_reg["date_registration"], errors="coerce")
    df_reg["date_unregistration"] = pd.to_numeric(df_reg["date_unregistration"], errors="coerce")

    # Chuẩn hóa chuỗi cho các cột phân loại (loại khoảng trắng thừa nếu có)
    for col in config.CATEGORICAL_FEATURES:
        if col in df_info.columns:
            df_info[col] = df_info[col].astype("string").str.strip()

    # Ghép 1-1 trên khóa chính (code_module, code_presentation, id_student)
    df = pd.merge(df_info, df_reg, on=config.PRIMARY_KEY, how="inner", validate="one_to_one")
    n_raw = len(df)
    print(f"  [Audit] Tổng lượt đăng ký gốc sau khi ghép 1-1: {n_raw:,} dòng")

    # Bước 1: Loại bản ghi mâu thuẫn giữa final_result và date_unregistration
    mask_inconsistent = (
        ((df["final_result"] == "Withdrawn") & df["date_unregistration"].isna()) |
        ((df["final_result"] != "Withdrawn") & df["date_unregistration"].notna())
    )
    n_step1 = int(mask_inconsistent.sum())
    df = df.loc[~mask_inconsistent].copy()
    print(f"  [Bước 1] Loại mâu thuẫn nhãn & ngày rút        : -{n_step1:,} dòng -> còn {len(df):,}")

    # Bước 2: Loại bản ghi thiếu date_registration
    mask_missing_reg = df["date_registration"].isna()
    n_step2 = int(mask_missing_reg.sum())
    df = df.loc[~mask_missing_reg].copy()
    print(f"  [Bước 2] Loại thiếu ngày đăng ký               : -{n_step2:,} dòng -> còn {len(df):,}")

    # Bước 3: Loại bản ghi đăng ký sau mốc ngày 28
    mask_late_reg = df["date_registration"] > config.CUTOFF_DAY
    n_step3 = int(mask_late_reg.sum())
    df = df.loc[~mask_late_reg].copy()
    print(f"  [Bước 3] Loại đăng ký muộn (date_reg > {config.CUTOFF_DAY})     : -{n_step3:,} dòng -> còn {len(df):,}")

    # Bước 4 (Chống Data Leakage): Loại các ca đã rút vào hoặc trước ngày 28
    mask_early_withdraw = df["date_unregistration"].notna() & (df["date_unregistration"] <= config.CUTOFF_DAY)
    n_step4 = int(mask_early_withdraw.sum())
    df = df.loc[~mask_early_withdraw].copy()
    print(f"  [Bước 4] Loại đã rút <= ngày {config.CUTOFF_DAY} (Anti-leakage) : -{n_step4:,} dòng -> còn {len(df):,}")

    # Định nghĩa nhãn mục tiêu nhị phân và số ngày tính từ lúc đăng ký đến ngày 28
    df[config.TARGET_COL] = df["final_result"].apply(
        lambda x: "Withdrawn" if x == "Withdrawn" else "Not_withdrawn"
    )
    df["days_since_registration"] = (config.CUTOFF_DAY - df["date_registration"]).astype(float)

    # Loại bỏ các cột gây rò rỉ tương lai khỏi file cohort chuẩn (chỉ giữ final_result để đối chiếu nếu cần)
    keep_cols = (
        config.PRIMARY_KEY
        + [c for c in config.CATEGORICAL_FEATURES if c not in config.PRIMARY_KEY]
        + ["num_of_prev_attempts", "studied_credits", "days_since_registration", "final_result", config.TARGET_COL]
    )
    df = df[keep_cols].reset_index(drop=True)

    # Xác thực toàn vẹn dữ liệu đầu ra
    assert not df.duplicated(subset=config.PRIMARY_KEY).any(), "Phát hiện trùng khóa chính trong cohort!"
    df.to_csv(config.COHORT_DATA_PATH, index=False)

    fig_path = os.path.join(config.FIGURES_DIR, "fig3_1_label_distribution.png")
    plot_label_distribution(df, fig_path)

    n_w = int((df[config.TARGET_COL] == "Withdrawn").sum())
    n_nw = int((df[config.TARGET_COL] == "Not_withdrawn").sum())
    print(f"✔ Đã xuất cohort_day28.csv: {len(df):,} dòng ({df['id_student'].nunique():,} SV) | "
          f"Not_withdrawn: {n_nw:,} ({n_nw/len(df)*100:.2f}%) | Withdrawn: {n_w:,} ({n_w/len(df)*100:.2f}%)")
    return df