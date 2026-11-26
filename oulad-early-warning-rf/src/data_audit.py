import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from src import config

plt.rcParams["axes.unicode_minus"] = False

RAW_TABLES = [
    "courses.csv",
    "assessments.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
]


def run_raw_audit(chunksize=500_000):
    """
    Quét trực tiếp 7/7 bảng gốc trong data/raw/ (tối ưu RAM cho studentVle.csv).
    Trả về dict chứa đầy đủ bảng tổng hợp, missing, phân vị outlier và số liệu phễu lọc 4 bước.
    """
    table_stats = []
    missing_rows = []

    # 1. Quét 6 bảng kích thước nhỏ/trung bình
    for fname in RAW_TABLES:
        path = os.path.join(config.RAW_DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)

        if fname == "studentVle.csv":
            n_rows = 0
            n_cols = 6
            miss_map = {}
            date_counts = {}
            click_max = 0
            click_gt_50 = 0

            for chunk in pd.read_csv(path, chunksize=chunksize, na_values=["?", ""]):
                n_cols = chunk.shape[1]
                n_rows += len(chunk)
                for c, v in chunk.isna().sum().items():
                    miss_map[c] = miss_map.get(c, 0) + int(v)
                vc = chunk["date"].dropna().astype(int).value_counts()
                for d, cnt in vc.items():
                    date_counts[d] = date_counts.get(d, 0) + int(cnt)
                c_max = int(chunk["sum_click"].max())
                if c_max > click_max:
                    click_max = c_max
                click_gt_50 += int((chunk["sum_click"] > 50).sum())

            total_miss = sum(miss_map.values())
            table_stats.append({
                "Bảng dữ liệu": fname,
                "Số dòng": n_rows,
                "Số cột": n_cols,
                "Dung lượng (MB)": round(size_mb, 2),
                "Ô khuyết (NaN/?)": total_miss,
                "Trùng nguyên dòng": "Đọc theo lô (chunk)",
            })
            for col, cnt in miss_map.items():
                if cnt > 0:
                    missing_rows.append({
                        "Bảng.Cột": f"studentVle.{col}",
                        "Số dòng khuyết": cnt,
                        "Tỷ lệ (%)": round(cnt / n_rows * 100, 2),
                    })
            vle_date_series = pd.Series(date_counts).sort_index()
            vle_stats = {
                "total_rows": n_rows,
                "pre_course_rows": int(vle_date_series[vle_date_series.index < 0].sum()),
                "in_window_rows": int(vle_date_series[(vle_date_series.index >= 0) & (vle_date_series.index <= config.CUTOFF_DAY)].sum()),
                "after_cutoff_rows": int(vle_date_series[vle_date_series.index > config.CUTOFF_DAY].sum()),
                "min_date": int(vle_date_series.index.min()),
                "max_date": int(vle_date_series.index.max()),
                "max_single_click": click_max,
                "clicks_gt_50": click_gt_50,
            }
        else:
            df = pd.read_csv(path, na_values=["?", ""])
            n_rows, n_cols = df.shape
            total_miss = int(df.isna().sum().sum())
            n_dup = int(df.duplicated().sum())
            table_stats.append({
                "Bảng dữ liệu": fname,
                "Số dòng": n_rows,
                "Số cột": n_cols,
                "Dung lượng (MB)": round(size_mb, 2),
                "Ô khuyết (NaN/?)": total_miss,
                "Trùng nguyên dòng": f"{n_dup:,}",
            })
            for col, cnt in df.isna().sum().items():
                if cnt > 0:
                    missing_rows.append({
                        "Bảng.Cột": f"{fname[:-4]}.{col}",
                        "Số dòng khuyết": int(cnt),
                        "Tỷ lệ (%)": round(cnt / n_rows * 100, 2),
                    })

    # 2. Kiểm toán tính nhất quán, rò rỉ dữ liệu & phễu lọc 4 bước
    df_info = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "studentInfo.csv"), na_values=["?", ""])
    df_reg = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "studentRegistration.csv"), na_values=["?", ""])
    df_reg["date_registration"] = pd.to_numeric(df_reg["date_registration"], errors="coerce")
    df_reg["date_unregistration"] = pd.to_numeric(df_reg["date_unregistration"], errors="coerce")

    df_merged = pd.merge(df_info, df_reg, on=config.PRIMARY_KEY, how="inner", validate="one_to_one")
    n_raw = len(df_merged)
    n_sv_raw = int(df_merged["id_student"].nunique())
    n_repeat_raw = n_raw - n_sv_raw
    n_imd_raw = int(df_merged["imd_band"].isna().sum())

    # Bước 1: Mâu thuẫn nhãn và ngày bỏ học
    m_w_nodate = (df_merged["final_result"] == "Withdrawn") & df_merged["date_unregistration"].isna()
    m_other_hasdate = (df_merged["final_result"] != "Withdrawn") & df_merged["date_unregistration"].notna()
    m_inconsistent = m_w_nodate | m_other_hasdate
    n_step1 = int(m_inconsistent.sum())
    df_s1 = df_merged.loc[~m_inconsistent]

    # Bước 2: Thiếu ngày đăng ký
    m_miss_reg = df_s1["date_registration"].isna()
    n_step2 = int(m_miss_reg.sum())
    df_s2 = df_s1.loc[~m_miss_reg]

    # Bước 3: Đăng ký muộn > ngày 28
    m_late_reg = df_s2["date_registration"] > config.CUTOFF_DAY
    n_step3 = int(m_late_reg.sum())
    df_s3 = df_s2.loc[~m_late_reg]

    # Bước 4: Đã bỏ học <= ngày 28 (Anti-Leakage)
    m_early_w = df_s3["date_unregistration"].notna() & (df_s3["date_unregistration"] <= config.CUTOFF_DAY)
    n_step4 = int(m_early_w.sum())
    df_clean = df_s3.loc[~m_early_w].copy()

    df_clean["target"] = np.where(df_clean["final_result"] == "Withdrawn", "Withdrawn", "Not_withdrawn")
    n_clean = len(df_clean)
    n_sv_clean = int(df_clean["id_student"].nunique())
    n_w_clean = int((df_clean["target"] == "Withdrawn").sum())
    n_nw_clean = n_clean - n_w_clean

    funnel = {
        "n_raw": n_raw,
        "n_sv_raw": n_sv_raw,
        "n_repeat_raw": n_repeat_raw,
        "n_imd_raw": n_imd_raw,
        "n_w_nodate": int(m_w_nodate.sum()),
        "n_other_hasdate": int(m_other_hasdate.sum()),
        "step1_removed": n_step1,
        "after_step1": len(df_s1),
        "step2_removed": n_step2,
        "after_step2": len(df_s2),
        "step3_removed": n_step3,
        "after_step3": len(df_s3),
        "step4_removed": n_step4,
        "n_clean": n_clean,
        "n_sv_clean": n_sv_clean,
        "n_w_clean": n_w_clean,
        "n_nw_clean": n_nw_clean,
        "n_imd_clean": int(df_clean["imd_band"].isna().sum()),
        "raw_final_result_counts": df_merged["final_result"].value_counts().to_dict(),
    }

    return {
        "df_tables": pd.DataFrame(table_stats),
        "df_missing": pd.DataFrame(missing_rows).sort_values("Tỷ lệ (%)", ascending=False) if missing_rows else pd.DataFrame(),
        "funnel": funnel,
        "vle_date_series": vle_date_series,
        "vle_stats": vle_stats,
        "df_clean_preview": df_clean,
    }


def plot_audit_dashboard(audit_res):
    """Vẽ Dashboard Kiểm toán 4 khung trực quan từ số liệu quét thực tế."""
    df_t = audit_res["df_tables"].sort_values("Số dòng", ascending=True)
    fn = audit_res["funnel"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6), dpi=160)
    fig.patch.set_facecolor("white")

    # 1. Quy mô 7 bảng gốc (thang log10)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(df_t["Bảng dữ liệu"], df_t["Số dòng"], color="#2A6F7F", height=0.62)
    ax1.set_xscale("log")
    ax1.set_xlim(10, max(df_t["Số dòng"]) * 25)
    ax1.set_title("1. Quy mô 7/7 bảng dữ liệu gốc (thang Log10)", fontweight="bold", color="#1B365D")
    ax1.set_xlabel("Số dòng bản ghi")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    for b, v in zip(bars1, df_t["Số dòng"]):
        ax1.text(v * 1.25, b.get_y() + b.get_height() / 2, f"{int(v):,}".replace(",", "."),
                 va="center", fontsize=8.5, fontweight="bold")

    # 2. Phễu lọc 4 bước
    ax2 = axes[0, 1]
    stages = [
        "Gốc (1–1)",
        f"Sau B1\n(-{fn['step1_removed']})",
        f"Sau B2\n(-{fn['step2_removed']})",
        f"Sau B3\n(-{fn['step3_removed']})",
        f"Cohort sạch\n(-{fn['step4_removed']:,})".replace(",", ".")
    ]
    vals = [fn["n_raw"], fn["after_step1"], fn["after_step2"], fn["after_step3"], fn["n_clean"]]
    cols = ["#1B365D", "#64748B", "#64748B", "#64748B", "#0F766E"]
    bars2 = ax2.bar(stages, vals, color=cols, width=0.55)
    ax2.set_ylim(0, fn["n_raw"] * 1.18)
    ax2.set_title(f"2. Phễu lọc 4 bước (Giữ {fn['n_clean']/fn['n_raw']*100:.2f}%)", fontweight="bold", color="#1B365D")
    ax2.set_ylabel("Số lượt đăng ký")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))
    for b, v in zip(bars2, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 600, f"{v:,}".replace(",", "."),
                 ha="center", fontsize=8.5, fontweight="bold")

    # 3. Các rủi ro chất lượng dữ liệu phát hiện
    ax3 = axes[1, 0]
    risk_labels = [
        "Đăng ký muộn (> ngày 28)",
        "Thiếu ngày đăng ký (NaN)",
        "Mâu thuẫn nhãn & ngày bỏ học",
        "Khuyết imd_band (tập gốc)",
        "Lặp mã SV đa học phần (gốc)",
        "Bỏ học ≤ ngày 28 (Leakage)",
    ]
    risk_vals = [
        fn["step3_removed"], fn["step2_removed"], fn["step1_removed"],
        fn["n_imd_raw"], fn["n_repeat_raw"], fn["step4_removed"]
    ]
    risk_cols = ["#64748B", "#D97706", "#B91C1C", "#D97706", "#0284C7", "#B91C1C"]
    bars3 = ax3.barh(risk_labels, risk_vals, color=risk_cols, height=0.6)
    ax3.set_xlim(0, max(risk_vals) * 1.3)
    ax3.set_title("3. Các vấn đề chất lượng dữ liệu phát hiện khi Audit", fontweight="bold", color="#1B365D")
    ax3.set_xlabel("Số lượt đăng ký bị ảnh hưởng")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    for b, v in zip(bars3, risk_vals):
        ax3.text(v + 80, b.get_y() + b.get_height() / 2, f"{v:,}".replace(",", "."),
                 va="center", fontsize=8.5, fontweight="bold")

    # 4. Phân bố final_result gốc vs nhãn nhị phân sau lọc
    ax4 = axes[1, 1]
    raw_fr = fn["raw_final_result_counts"]
    fr_keys = ["Pass", "Withdrawn", "Fail", "Distinction"]
    fr_vals = [raw_fr.get(k, 0) for k in fr_keys]
    fr_cols = ["#0F766E", "#B91C1C", "#D97706", "#0284C7"]
    bars4 = ax4.bar(fr_keys, fr_vals, color=fr_cols, width=0.52)
    ax4.set_ylim(0, max(fr_vals) * 1.2)
    ax4.set_title("4. Phân bố nhãn final_result gốc (32.593 lượt)", fontweight="bold", color="#1B365D")
    ax4.set_ylabel("Số lượt đăng ký")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars4, fr_vals):
        ax4.text(b.get_x() + b.get_width() / 2, v + 250, f"{v:,}\n({v/fn['n_raw']*100:.1f}%)".replace(",", "."),
                 ha="center", fontsize=8.5, fontweight="bold")

    fig.tight_layout(pad=1.8)
    return fig


def plot_vle_timeline(vle_date_series):
    """Biểu đồ phân bố nhật ký tương tác VLE theo ngày, đánh dấu vùng <0, 0-28 và >28."""
    fig, ax = plt.subplots(figsize=(11.5, 3.8), dpi=160)
    dates = vle_date_series.index.values
    counts = vle_date_series.values

    colors = np.where(dates < 0, "#0284C7", np.where(dates <= config.CUTOFF_DAY, "#0F766E", "#CBD5E1"))
    ax.bar(dates, counts, color=colors, width=1.0)
    ax.axvline(0, color="#1B365D", linestyle="--", lw=1.4, label="Ngày khai giảng (t = 0)")
    ax.axvline(config.CUTOFF_DAY, color="#B91C1C", linestyle="--", lw=1.8, label=f"Mốc cắt Anti-Leakage (t = {config.CUTOFF_DAY})")

    ax.set_title("Phân bố dòng log studentVle.csv theo ngày — Tách vùng Pre-course (<0), 0–28 và Vùng cắt bỏ (>28)",
                 fontweight="bold", color="#1B365D")
    ax.set_xlabel("Ngày tương đối so với khai giảng (date)")
    ax.set_ylabel("Số bản ghi tương tác")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_dropout_rate_by_category(df_clean, cat_col, title_vi):
    """Biểu đồ tỷ lệ Bỏ học (%) và quy mô sinh viên theo từng nhóm phân loại."""
    tmp = df_clean.copy()
    tmp[cat_col] = tmp[cat_col].fillna("Unknown")
    grp = tmp.groupby(cat_col).agg(
        total=("target", "size"),
        withdrawn=("target", lambda s: (s == "Withdrawn").sum())
    ).reset_index()
    grp["dropout_rate"] = grp["withdrawn"] / grp["total"] * 100
    grp = grp.sort_values("dropout_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, max(3.6, len(grp) * 0.42)), dpi=160)
    bars = ax.barh(grp[cat_col].astype(str), grp["dropout_rate"], color="#3B6978", height=0.62)
    avg_rate = (tmp["target"] == "Withdrawn").mean() * 100
    ax.axvline(avg_rate, color="#B91C1C", linestyle="--", lw=1.4, label=f"Trung bình toàn tập ({avg_rate:.2f}%)")

    ax.set_xlim(0, max(grp["dropout_rate"].max() * 1.28, 30))
    ax.set_xlabel("Tỷ lệ Bỏ học (%)")
    ax.set_title(f"Tỷ lệ Bỏ học (Withdrawn) theo {title_vi}", fontweight="bold", color="#1B365D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")

    for b, (_, row) in zip(bars, grp.iterrows()):
        ax.text(row["dropout_rate"] + 0.6, b.get_y() + b.get_height() / 2,
                f"{row['dropout_rate']:.1f}% (n={int(row['total']):,})".replace(",", "."),
                va="center", fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_behavior_comparison(df_feat):
    """So sánh phân phối hành vi VLE & bài tập sớm giữa nhóm Không bỏ học vs Bỏ học."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=160)
    w_df = df_feat[df_feat[config.TARGET_COL] == "Withdrawn"]
    nw_df = df_feat[df_feat[config.TARGET_COL] == "Not_withdrawn"]

    # 1. Log1p(total_clicks)
    axes[0].hist(np.log1p(nw_df["total_clicks"]), bins=30, alpha=0.65, density=True, color="#64748B", label="Không bỏ học")
    axes[0].hist(np.log1p(w_df["total_clicks"]), bins=30, alpha=0.65, density=True, color="#B91C1C", label="Bỏ học")
    axes[0].set_title("Phân phối log(1 + total_clicks)", fontweight="bold", color="#1B365D")
    axes[0].set_xlabel("log1p(total_clicks)")
    axes[0].legend(frameon=False, fontsize=8.5)

    # 2. recency_days
    axes[1].hist(nw_df["recency_days"], bins=30, alpha=0.65, density=True, color="#64748B", label="Không bỏ học")
    axes[1].hist(w_df["recency_days"], bins=30, alpha=0.65, density=True, color="#B91C1C", label="Bỏ học")
    axes[1].set_title("Số ngày ngưng học sát t=28 (recency_days)", fontweight="bold", color="#1B365D")
    axes[1].set_xlabel("recency_days (29 = chưa click lần nào)")
    axes[1].legend(frameon=False, fontsize=8.5)

    # 3. Nhịp click trung bình 4 tuần
    weeks = ["Tuần 1\n(0–6)", "Tuần 2\n(7–13)", "Tuần 3\n(14–20)", "Tuần 4\n(21–28)"]
    w_cols = ["clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28"]
    nw_means = [nw_df[c].mean() for c in w_cols]
    w_means = [w_df[c].mean() for c in w_cols]
    axes[2].plot(weeks, nw_means, marker="o", lw=2.2, color="#0F766E", label="Không bỏ học")
    axes[2].plot(weeks, w_means, marker="s", lw=2.2, color="#B91C1C", label="Bỏ học")
    axes[2].set_title("Xu hướng số click trung bình qua 4 tuần", fontweight="bold", color="#1B365D")
    axes[2].set_ylabel("Click trung bình / SV")
    axes[2].legend(frameon=False, fontsize=8.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df_feat):
    """Heatmap tương quan giữa các biến số cốt lõi (giải thích đa cộng tuyến)."""
    candidate_cols = [
        "total_clicks", "active_days", "unique_sites", "recency_days",
        "clicks_d0_6", "clicks_d21_28", "pre_course_clicks",
        "click_trend_w4_vs_w1", "zero_click_weeks",
        "num_submitted_d28", "mean_score_d28", "missing_assessments_d28"
    ]
    cols = [c for c in candidate_cols if c in df_feat.columns]
    corr = df_feat[cols].corr()

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=160)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(cols, fontsize=8.5)
    ax.set_title("Ma trận tương quan Pearson các đặc trưng hành vi & bài tập cốt lõi", fontweight="bold", color="#1B365D", pad=12)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.2, color="white" if abs(val) > 0.55 else "#0F172A")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig