"""Xây dựng 39 đặc trưng tại ngày dự báo (CUTOFF_DAY) cho nhóm sinh viên trong cohort.
 
Nguyên tắc chống rò rỉ dữ liệu (leakage):
  - Chỉ dùng nhật ký VLE có date <= CUTOFF_DAY.
  - Không dùng date_unregistration, final_result (chỉ giữ cột `target` làm nhãn).
 
Định nghĩa các đặc trưng (khớp với src/config.py):
  days_since_registration : CUTOFF_DAY - date_registration
  total_clicks            : tổng sum_click, mọi ngày <= CUTOFF_DAY (gồm cả trước khi khóa bắt đầu)
  active_days             : số ngày khác nhau có click
  unique_sites            : số id_site khác nhau đã truy cập
  recency_days            : CUTOFF_DAY - ngày hoạt động gần nhất
                            (sinh viên không có hoạt động -> NO_ACTIVITY_RECENCY, xem bên dưới)
  mean_clicks_active_day  : total_clicks / active_days (0 nếu không có hoạt động)
  clicks_d0_6 ... d21_28  : click theo 4 cửa sổ tuần (ngày 0-6, 7-13, 14-20, 21-28)
  activity_clicks_<loại>  : tổng click theo loại tài nguyên VLE (20 loại)
"""
import os
 
import pandas as pd
 
from src import config
 
KEY = config.PRIMARY_KEY
WEEK_WINDOWS = {"clicks_d0_6": (0, 6), "clicks_d7_13": (7, 13),
                "clicks_d14_20": (14, 20), "clicks_d21_28": (21, 28)}
 
 
def _load_cohort():
    if not os.path.exists(config.COHORT_PATH):
        from src.cohort_builder import build_cohort
        return build_cohort()
    return pd.read_csv(config.COHORT_PATH)
 
 
def build_features(cohort=None, save=True, verbose=True):
    """Trả về DataFrame: PRIMARY_KEY + ALL_FEATURES + TARGET_COL (1 dòng / lượt đăng ký)."""
    cohort = _load_cohort() if cohort is None else cohort
    cutoff = config.CUTOFF_DAY
 
    vle = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "vle.csv"), usecols=["id_site", "activity_type"])
    log = pd.read_csv(os.path.join(config.RAW_DATA_DIR, "studentVle.csv"),
                      usecols=KEY + ["id_site", "date", "sum_click"])
    log = log[log["date"] <= cutoff]
    log = log.merge(cohort[KEY], on=KEY, how="inner")           # chỉ giữ sinh viên thuộc cohort
    log["activity_type"] = log["id_site"].map(vle.set_index("id_site")["activity_type"])
    if verbose:
        print(f"Nhật ký VLE dùng cho đặc trưng: {len(log):,} dòng (date <= {cutoff})")
 
    g = log.groupby(KEY)
    agg = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
        "unique_sites": g["id_site"].nunique(),
        "last_date": g["date"].max(),
    })
    agg["recency_days"] = cutoff - agg["last_date"]
    agg = agg.drop(columns="last_date")
 
    for name, (lo, hi) in WEEK_WINDOWS.items():
        m = log["date"].between(lo, hi)
        agg = agg.join(log[m].groupby(KEY)["sum_click"].sum().rename(name), how="left")
 
    act = (log.pivot_table(index=KEY, columns="activity_type", values="sum_click",
                           aggfunc="sum", fill_value=0)
              .reindex(columns=config.ACTIVITY_TYPES, fill_value=0)
              .add_prefix("activity_clicks_"))
    agg = agg.join(act, how="left")
 
    df = cohort.merge(agg.reset_index(), on=KEY, how="left")
 
    # Sinh viên không có nhật ký nào trong cửa sổ = 0 hoạt động (không phải thiếu dữ liệu)
    zero_cols = ["total_clicks", "active_days", "unique_sites"] + list(WEEK_WINDOWS) + config.ACTIVITY_FEATURES
    df[zero_cols] = df[zero_cols].fillna(0)
 
    # recency của người không có hoạt động: lớn hơn mọi giá trị thực tế (lần hoạt động sớm nhất trong dữ liệu + 1)
    no_activity_recency = int(cutoff - log["date"].min() + 1)
    df["recency_days"] = df["recency_days"].fillna(no_activity_recency)
 
    df["mean_clicks_active_day"] = (df["total_clicks"] / df["active_days"].where(df["active_days"] > 0)).fillna(0)
    df["days_since_registration"] = cutoff - df["date_registration"]
 
    out = df[KEY + config.ALL_FEATURES + [config.TARGET_COL]].copy()
    if verbose:
        miss = out[config.ALL_FEATURES].isna().sum()
        miss = miss[miss > 0]
        print(f"Bảng đặc trưng: {out.shape[0]:,} dòng x {len(config.ALL_FEATURES)} đặc trưng")
        print("Cột còn thiếu giá trị (sẽ được impute trong pipeline):", miss.to_dict() or "không có")
    if save:
        out.to_csv(config.FEATURES_PATH, index=False)
        if verbose:
            print(f"Đã lưu: {config.FEATURES_PATH}")
    return out
 
 
def load_features():
    """Đọc bảng đặc trưng đã lưu; nếu chưa có thì tự xây."""
    if os.path.exists(config.FEATURES_PATH):
        return pd.read_csv(config.FEATURES_PATH)
    return build_features()
 
 
if __name__ == "__main__":
    build_features()