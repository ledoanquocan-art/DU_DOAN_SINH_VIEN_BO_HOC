import numpy as np
import pandas as pd
from src import config

# Nhóm cột lõi tối thiểu cần có để đảm bảo độ tin cậy dự đoán
CORE_REQUIRED_FEATURES = [
    "code_module",
    "clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28",
    "active_days", "recency_days"
]


def prepare_unlabeled_features(df_raw, model):
    """
    Chuẩn hóa file CSV rút gọn đầu vào cho tính năng số 5:
      1. Tự tính lại các cột dẫn xuất (total_clicks, mean_clicks_active_day, 5 cột trend) nếu thiếu.
      2. Điền 0 cho các cột activity_clicks_* bị thiếu (đúng bản chất log VLE).
      3. Điền NaN cho các cột ít quan trọng còn lại để SimpleImputer trong Pipeline tự xử lý.
    """
    expected_cols = (
        list(model.feature_names_in_)
        if hasattr(model, "feature_names_in_")
        else list(config.ALL_FEATURES)
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