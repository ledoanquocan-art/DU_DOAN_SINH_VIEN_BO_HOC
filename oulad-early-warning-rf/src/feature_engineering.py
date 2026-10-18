import os
import hashlib
import numpy as np
import pandas as pd
from src import config

def generate_features():
    """
    Tổng hợp hành vi VLE trong cửa sổ ngày 0-28 và tạo bảng 39 đặc trưng (features_day28.csv).
    """
    cohort_path = os.path.join(config.PROCESSED_DATA_DIR, "cohort_day28.csv")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(
            "Chưa có file cohort_day28.csv! Vui lòng chạy build_cohort() trước."
        )

    vle_path = os.path.join(config.RAW_DATA_DIR, "vle.csv")
    student_vle_path = os.path.join(config.RAW_DATA_DIR, "studentVle.csv")
    if not os.path.exists(vle_path) or not os.path.exists(student_vle_path):
        raise FileNotFoundError(
            "Thiếu file vle.csv hoặc studentVle.csv trong thư mục data/raw/!"
        )

    base = pd.read_csv(cohort_path)
    key = config.PRIMARY_KEY
    cutoff = config.CUTOFF_DAY

    # Đọc metadata tài nguyên VLE (kiểm tra khóa duy nhất để tránh nhân bản lượt click)
    vle = pd.read_csv(
        vle_path,
        usecols=["code_module", "code_presentation", "id_site", "activity_type"]
    )

    print("  ⏳ Đang đọc studentVle.csv theo khối (chunksize=500,000) và lọc 0 <= date <= 28...")
    valid_keys = base[key].drop_duplicates()

    chunks = []
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    dtypes = {
        "code_module": "string",
        "code_presentation": "string",
        "id_student": "int64",
        "id_site": "int64",
        "date": "int16",
        "sum_click": "int32"
    }

    for chunk in pd.read_csv(student_vle_path, usecols=usecols, dtype=dtypes, chunksize=500_000):
        # Chỉ giữ nhật ký trong cửa sổ từ ngày 0 đến ngày 28 (bao gồm hai đầu)
        sub = chunk.loc[chunk["date"].between(0, cutoff)]
        if not sub.empty:
            sub = sub.merge(valid_keys, on=key, how="inner", validate="many_to_one")
            if not sub.empty:
                chunks.append(sub)

    if chunks:
        logs = pd.concat(chunks, ignore_index=True)
    else:
        logs = pd.DataFrame(columns=usecols)

    # Ghép metadata vle theo (code_module, code_presentation, id_site)
    logs = logs.merge(
        vle,
        on=["code_module", "code_presentation", "id_site"],
        how="left",
        validate="many_to_one"
    )

    # 1. Tổng hợp mức độ hoạt động chung (5 biến)
    feat_general = logs.groupby(key, as_index=False).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        unique_sites=("id_site", "nunique"),
        last_day=("date", "max")
    )

    base = base.merge(feat_general, on=key, how="left", validate="one_to_one")
    for col in ["total_clicks", "active_days", "unique_sites"]:
        base[col] = base[col].fillna(0).astype(int)

    # recency_days = 28 - ngày hoạt động cuối; gán 29 nếu không có hoạt động trong cửa sổ
    base["recency_days"] = (cutoff - base["last_day"].fillna(-1)).astype(int)

    # mean_clicks_active_day = total_clicks / active_days; gán 0 khi active_days = 0
    base["mean_clicks_active_day"] = (
        base["total_clicks"] / base["active_days"].replace(0, np.nan)
    ).fillna(0.0)

    # days_since_registration = 28 - date_registration
    base["days_since_registration"] = (cutoff - base["date_registration"]).astype(float)

    # 2. Tổng hợp tổng click theo 4 khoảng ngày: 0-6, 7-13, 14-20, 21-28
    windows = [(0, 6), (7, 13), (14, 20), (21, 28)]
    for start_d, end_d in windows:
        col_name = f"clicks_d{start_d}_{end_d}"
        win_df = (
            logs.loc[logs["date"].between(start_d, end_d)]
            .groupby(key, as_index=False)["sum_click"]
            .sum()
            .rename(columns={"sum_click": col_name})
        )
        base = base.merge(win_df, on=key, how="left", validate="one_to_one")
        base[col_name] = base[col_name].fillna(0).astype(int)

    # 3. Tổng hợp 20 biến tổng click riêng theo activity_type
    act_pivot = logs.pivot_table(
        index=key,
        columns="activity_type",
        values="sum_click",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    rename_map = {act: f"activity_clicks_{act}" for act in config.ACTIVITY_TYPES}
    act_pivot = act_pivot.rename(columns=rename_map)

    # Đảm bảo đủ cả 20 cột activity_clicks_* ngay cả khi có loại bằng 0 toàn bộ
    for act_col in config.ACTIVITY_FEATURES:
        if act_col not in act_pivot.columns:
            act_pivot[act_col] = 0

    base = base.merge(
        act_pivot[key + config.ACTIVITY_FEATURES],
        on=key,
        how="left",
        validate="one_to_one"
    )
    for act_col in config.ACTIVITY_FEATURES:
        base[act_col] = base[act_col].fillna(0).astype(int)

    # Sắp xếp cột chuẩn: Khóa quản lý + 39 đặc trưng đầu vào + Nhãn mục tiêu
    output_cols = key + config.ALL_FEATURES + [config.TARGET_COL]
    final_df = base[output_cols].copy()

    out_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
    final_df.to_csv(out_path, index=False)

    # Thống kê kiểm tra khớp số liệu báo cáo
    zero_click_rows = int((final_df["total_clicks"] == 0).sum())
    missing_imd = int(final_df["imd_band"].isna().sum())
    sha256_hash = hashlib.sha256(open(out_path, "rb").read()).hexdigest()

    print("\n" + "="*60)
    print("✅ HOÀN TẤT TRÍCH XUẤT BẢNG 39 ĐẶC TRƯNG TẠI NGÀY 28")
    print("="*60)
    print(f"  • Kích thước bảng dữ liệu     : {len(final_df):,} dòng x {len(config.ALL_FEATURES)} đặc trưng")
    print(f"  • Số lượt đăng ký 0 click     : {zero_click_rows:,} dòng (vẫn giữ trong nhóm dự báo)")
    print(f"  • Số dòng thiếu imd_band      : {missing_imd:,} dòng (sẽ gán 'Unknown' trong Pipeline)")
    print(f"  • Đường dẫn file xuất         : {out_path}")
    print(f"  • SHA-256 features_day28.csv  : {sha256_hash}")
    print("="*60)

    return final_df