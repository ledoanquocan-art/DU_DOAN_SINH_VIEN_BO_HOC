import os
import numpy as np
import pandas as pd
from src import config


def extract_vle_features(cohort_keys, chunksize=500_000):
    """
    Đọc studentVle.csv theo từng khối (chunk) để tiết kiệm RAM.
    Lọc date <= 28 (chống rò rỉ tương lai), tách riêng:
      - Tương tác trước khai giảng (date < 0): pre_course_clicks, pre_course_active_days
      - Tương tác trong 28 ngày đầu (0 <= date <= 28): 12 biến tổng hợp + 20 loại tài nguyên VLE
    """
    vle_path = os.path.join(config.RAW_DATA_DIR, "vle.csv")
    student_vle_path = os.path.join(config.RAW_DATA_DIR, "studentVle.csv")

    df_vle = pd.read_csv(vle_path, usecols=["id_site", "activity_type"], na_values=["?", ""])
    site_to_activity = df_vle.drop_duplicates("id_site").set_index("id_site")["activity_type"]

    # Các bộ gom trung gian xuyên suốt các chunk
    pre_clicks_list = []
    pre_dates_list = []

    in_clicks_list = []
    in_dates_list = []
    in_sites_list = []
    in_activity_list = []

    pk = config.PRIMARY_KEY
    usecols = pk + ["id_site", "date", "sum_click"]

    print("  ⏳ Đang đọc và tổng hợp studentVle.csv (lọc date <= 28 theo từng khối 500k dòng)...")
    reader = pd.read_csv(student_vle_path, usecols=usecols, chunksize=chunksize, na_values=["?", ""])

    for chunk in reader:
        # Khóa chặt mốc thời gian: tuyệt đối không lấy dữ liệu sau ngày 28
        chunk = chunk[chunk["date"] <= config.CUTOFF_DAY]
        if chunk.empty:
            continue

        # Chỉ giữ lại các lượt đăng ký thuộc nhóm Cohort hợp lệ
        chunk = chunk.merge(cohort_keys, on=pk, how="inner")
        if chunk.empty:
            continue

        # --- PHẦN A: Tương tác TRƯỚC ngày khai giảng (date < 0) ---
        chunk_pre = chunk[chunk["date"] < 0]
        if not chunk_pre.empty:
            pre_clicks_list.append(
                chunk_pre.groupby(pk, sort=False)["sum_click"].sum()
            )
            pre_dates_list.append(
                chunk_pre[pk + ["date"]].drop_duplicates()
            )

        # --- PHẦN B: Tương tác TRONG 28 ngày đầu (0 <= date <= 28) ---
        chunk_in = chunk[chunk["date"] >= 0].copy()
        if not chunk_in.empty:
            # Tách số click theo 4 tuần
            d = chunk_in["date"]
            sc = chunk_in["sum_click"]
            chunk_in["clicks_d0_6"] = np.where((d >= 0) & (d <= 6), sc, 0)
            chunk_in["clicks_d7_13"] = np.where((d >= 7) & (d <= 13), sc, 0)
            chunk_in["clicks_d14_20"] = np.where((d >= 14) & (d <= 20), sc, 0)
            chunk_in["clicks_d21_28"] = np.where((d >= 21) & (d <= 28), sc, 0)

            in_clicks_list.append(
                chunk_in.groupby(pk, sort=False)[
                    ["sum_click", "clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28"]
                ].sum()
            )
            in_dates_list.append(chunk_in[pk + ["date"]].drop_duplicates())
            in_sites_list.append(chunk_in[pk + ["id_site"]].drop_duplicates())

            # Gom click theo 20 loại học liệu VLE
            chunk_in["activity_type"] = chunk_in["id_site"].map(site_to_activity)
            chunk_act = chunk_in.dropna(subset=["activity_type"])
            if not chunk_act.empty:
                in_activity_list.append(
                    chunk_act.groupby(pk + ["activity_type"], sort=False)["sum_click"].sum()
                )

    # Khởi tạo khung kết quả chuẩn theo cohort_keys
    vle_feat = cohort_keys.copy()

    # 1. Tổng hợp đặc trưng trước khai giảng (Pre-course)
    if pre_clicks_list:
        pre_clicks = pd.concat(pre_clicks_list).groupby(level=[0, 1, 2]).sum().rename("pre_course_clicks")
        vle_feat = vle_feat.merge(pre_clicks.reset_index(), on=pk, how="left")
    else:
        vle_feat["pre_course_clicks"] = 0

    if pre_dates_list:
        pre_dates = pd.concat(pre_dates_list, ignore_index=True).drop_duplicates()
        pre_active = pre_dates.groupby(pk)["date"].nunique().rename("pre_course_active_days")
        vle_feat = vle_feat.merge(pre_active.reset_index(), on=pk, how="left")
    else:
        vle_feat["pre_course_active_days"] = 0

    # 2. Tổng hợp click tổng và click 4 tuần trong giai đoạn ngày 0-28
    if in_clicks_list:
        in_clicks = pd.concat(in_clicks_list).groupby(level=[0, 1, 2]).sum()
        in_clicks = in_clicks.rename(columns={"sum_click": "total_clicks"})
        vle_feat = vle_feat.merge(in_clicks.reset_index(), on=pk, how="left")
    else:
        for col in ["total_clicks", "clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28"]:
            vle_feat[col] = 0

    # 3. Tổng hợp active_days và recency_days (số ngày ngưng tương tác tính đến ngày 28)
    if in_dates_list:
        all_in_dates = pd.concat(in_dates_list, ignore_index=True).drop_duplicates()
        date_stats = all_in_dates.groupby(pk)["date"].agg(
            active_days="nunique",
            last_active_day="max"
        ).reset_index()
        date_stats["recency_days"] = config.CUTOFF_DAY - date_stats["last_active_day"]
        vle_feat = vle_feat.merge(date_stats[pk + ["active_days", "recency_days"]], on=pk, how="left")
    else:
        vle_feat["active_days"] = 0
        vle_feat["recency_days"] = config.CUTOFF_DAY + 1

    # 4. Tổng hợp unique_sites
    if in_sites_list:
        all_in_sites = pd.concat(in_sites_list, ignore_index=True).drop_duplicates()
        site_stats = all_in_sites.groupby(pk)["id_site"].nunique().rename("unique_sites")
        vle_feat = vle_feat.merge(site_stats.reset_index(), on=pk, how="left")
    else:
        vle_feat["unique_sites"] = 0

    # 5. Tổng hợp 20 cột activity_clicks_*
    if in_activity_list:
        act_sum = pd.concat(in_activity_list).groupby(level=[0, 1, 2, 3]).sum()
        act_pivot = act_sum.unstack(fill_value=0)
        act_pivot.columns = [f"activity_clicks_{c}" for c in act_pivot.columns]
        vle_feat = vle_feat.merge(act_pivot.reset_index(), on=pk, how="left")

    for act_col in config.ACTIVITY_FEATURES:
        if act_col not in vle_feat.columns:
            vle_feat[act_col] = 0

    # Điền 0 cho các sinh viên hoàn toàn không có log VLE
    zero_fill_cols = (
        ["pre_course_clicks", "pre_course_active_days", "total_clicks", "active_days", "unique_sites",
         "clicks_d0_6", "clicks_d7_13", "clicks_d14_20", "clicks_d21_28"]
        + config.ACTIVITY_FEATURES
    )
    for col in zero_fill_cols:
        vle_feat[col] = vle_feat[col].fillna(0).astype(int)

    # Nếu chưa click ngày nào trong 0..28 thì recency_days = 29 (quy ước hệ thống)
    vle_feat["recency_days"] = vle_feat["recency_days"].fillna(config.CUTOFF_DAY + 1).astype(int)

    # Trung bình số click mỗi ngày có tương tác
    vle_feat["mean_clicks_active_day"] = np.where(
        vle_feat["active_days"] > 0,
        (vle_feat["total_clicks"] / vle_feat["active_days"]).round(4),
        0.0
    )

    return vle_feat


def add_trend_features(df):
    """
    Tạo 5 đặc trưng đo xu hướng suy giảm và tính ổn định nhịp học trong 4 tuần đầu.
    """
    w1 = df["clicks_d0_6"].astype(float)
    w2 = df["clicks_d7_13"].astype(float)
    w3 = df["clicks_d14_20"].astype(float)
    w4 = df["clicks_d21_28"].astype(float)

    # Tỷ lệ biến động tuần 4 so với tuần 1 & tuần 3 (cộng 1 làm trơn tránh chia cho 0)
    df["click_trend_w4_vs_w1"] = ((w4 + 1.0) / (w1 + 1.0)).round(4)
    df["click_trend_w4_vs_w3"] = ((w4 + 1.0) / (w3 + 1.0)).round(4)

    # Tổng số tuần có 0 click trong 4 tuần đầu (0..4)
    weekly_matrix = np.column_stack([w1, w2, w3, w4])
    df["zero_click_weeks"] = (weekly_matrix == 0).sum(axis=1).astype(int)

    # Số tuần liên tiếp gần nhất (tính ngược từ tuần 4 về tuần 1) bị 0 click
    df["trailing_zero_weeks"] = np.select(
        [w4 > 0, w3 > 0, w2 > 0, w1 > 0],
        [0, 1, 2, 3],
        default=4
    ).astype(int)

    # Độ lệch chuẩn số click qua 4 tuần (đo tính đều đặn vs học dồn)
    df["click_std_4w"] = np.std(weekly_matrix, axis=1).round(4)

    return df


def extract_assessment_features(cohort_keys):
    """
    Trích xuất 4 đặc trưng bài tập & điểm số sớm tại mốc ngày 28 (t <= 28):
      - num_submitted_d28: Số bài kiểm tra đã nộp tính đến ngày 28
      - mean_score_d28: Điểm trung bình các bài đã nộp đến ngày 28 (NaN nếu chưa nộp)
      - missing_assessments_d28: Số bài có hạn nộp <= 28 nhưng sinh viên chưa nộp đến ngày 28
      - avg_submission_delay_d28: Độ trễ nộp bài trung bình (âm = nộp sớm, dương = nộp muộn)
    """
    print("  ⏳ Đang trích xuất đặc trưng bài tập & điểm số sớm (assessments <= ngày 28)...")
    assess_path = os.path.join(config.RAW_DATA_DIR, "assessments.csv")
    stu_assess_path = os.path.join(config.RAW_DATA_DIR, "studentAssessment.csv")

    df_assess = pd.read_csv(assess_path, na_values=["?", ""])
    df_stu_assess = pd.read_csv(stu_assess_path, na_values=["?", ""])

    df_assess["date"] = pd.to_numeric(df_assess["date"], errors="coerce")
    df_stu_assess["date_submitted"] = pd.to_numeric(df_stu_assess["date_submitted"], errors="coerce")
    df_stu_assess["score"] = pd.to_numeric(df_stu_assess["score"], errors="coerce")
    df_stu_assess["is_banked"] = pd.to_numeric(df_stu_assess["is_banked"], errors="coerce").fillna(0)

    # Ghép studentAssessment với assessments qua id_assessment để gắn (code_module, code_presentation)
    sa = pd.merge(
        df_stu_assess,
        df_assess[["code_module", "code_presentation", "id_assessment", "assessment_type", "date"]],
        on="id_assessment",
        how="inner"
    )

    pk = config.PRIMARY_KEY
    # Khóa chống rò rỉ: chỉ lấy các bài đã nộp vào hoặc trước ngày 28 và thuộc nhóm Cohort
    sa_d28 = sa[sa["date_submitted"].notna() & (sa["date_submitted"] <= config.CUTOFF_DAY)].copy()
    sa_d28 = sa_d28.merge(cohort_keys, on=pk, how="inner")

    # 1. Tổng số bài đã nộp & điểm trung bình các bài đã nộp đến ngày 28
    sub_stats = sa_d28.groupby(pk).agg(
        num_submitted_d28=("id_assessment", "nunique"),
        mean_score_d28=("score", "mean")
    ).reset_index()
    sub_stats["mean_score_d28"] = sub_stats["mean_score_d28"].round(2)

    # 2. Độ trễ nộp bài trung bình (loại bài banked từ kỳ trước và bài không có deadline)
    sa_delay = sa_d28[(sa_d28["is_banked"] == 0) & sa_d28["date"].notna()].copy()
    sa_delay["delay"] = sa_delay["date_submitted"] - sa_delay["date"]
    delay_stats = sa_delay.groupby(pk)["delay"].mean().round(2).rename("avg_submission_delay_d28").reset_index()

    # 3. Số bài BỎ NỘP trong số các bài có hạn nộp <= ngày 28
    due_d28 = df_assess[df_assess["date"].notna() & (df_assess["date"] <= config.CUTOFF_DAY)]
    due_count_by_course = (
        due_d28.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique()
        .rename("due_assessments_d28")
        .reset_index()
    )

    sa_due_submitted = sa_d28[sa_d28["date"].notna() & (sa_d28["date"] <= config.CUTOFF_DAY)]
    submitted_due_stats = (
        sa_due_submitted.groupby(pk)["id_assessment"]
        .nunique()
        .rename("submitted_due_d28")
        .reset_index()
    )

    # Hợp nhất vào khung cohort_keys
    assess_feat = cohort_keys.merge(due_count_by_course, on=["code_module", "code_presentation"], how="left")
    assess_feat = assess_feat.merge(submitted_due_stats, on=pk, how="left")
    assess_feat = assess_feat.merge(sub_stats, on=pk, how="left")
    assess_feat = assess_feat.merge(delay_stats, on=pk, how="left")

    assess_feat["due_assessments_d28"] = assess_feat["due_assessments_d28"].fillna(0).astype(int)
    assess_feat["submitted_due_d28"] = assess_feat["submitted_due_d28"].fillna(0).astype(int)
    assess_feat["num_submitted_d28"] = assess_feat["num_submitted_d28"].fillna(0).astype(int)
    assess_feat["missing_assessments_d28"] = (
        assess_feat["due_assessments_d28"] - assess_feat["submitted_due_d28"]
    ).clip(lower=0).astype(int)

    # Giữ nguyên NaN ở mean_score_d28 & avg_submission_delay_d28 để SimpleImputer xử lý sạch trong Pipeline CV
    return assess_feat[pk + config.ASSESSMENT_FEATURES]


def generate_features():
    """
    Hợp nhất toàn bộ 50 đặc trưng đầu vào (7 phân loại + 12 cơ bản + 20 VLE + 11 cải tiến)
    và lưu ra data/processed/features_day28.csv.
    """
    if not os.path.exists(config.COHORT_DATA_PATH):
        raise FileNotFoundError(
            "Chưa có file cohort_day28.csv! Vui lòng chạy build_cohort() trước."
        )

    df_cohort = pd.read_csv(config.COHORT_DATA_PATH, na_values=["?", ""])
    pk = config.PRIMARY_KEY
    cohort_keys = df_cohort[pk].copy()

    # 1. Trích xuất VLE (cả pre-course < 0 và in-course 0..28)
    vle_feat = extract_vle_features(cohort_keys)

    # 2. Tạo 5 đặc trưng xu hướng suy giảm nhịp học
    vle_feat = add_trend_features(vle_feat)

    # 3. Trích xuất 4 đặc trưng bài tập & điểm số sớm (<= ngày 28)
    assess_feat = extract_assessment_features(cohort_keys)

    # 4. Ghép toàn bộ đặc trưng vào bảng Cohort
    df_final = df_cohort.merge(vle_feat, on=pk, how="left", validate="one_to_one")
    df_final = df_final.merge(assess_feat, on=pk, how="left", validate="one_to_one")

    # Sắp xếp cột chuẩn hóa: Khóa định danh -> 50 đặc trưng -> nhãn
    meta_cols = [c for c in ["code_presentation", "id_student"] if c not in config.ALL_FEATURES]
    extra_cols = [c for c in ["final_result", config.TARGET_COL] if c in df_final.columns]
    ordered_cols = meta_cols + config.ALL_FEATURES + extra_cols
    df_final = df_final[ordered_cols]

    # Kiểm tra không phát sinh giá trị vô cực (inf) ở các phép chia tỷ lệ
    num_vals = df_final[config.NUMERICAL_FEATURES].to_numpy(dtype=float)
    assert not np.isinf(num_vals).any(), "Phát hiện giá trị vô cực (inf) trong bảng đặc trưng!"
    assert len(df_final) == len(df_cohort), "Số dòng bị thay đổi sau khi ghép đặc trưng!"

    df_final.to_csv(config.FEATURES_DATA_PATH, index=False)
    print(f"✔ Hoàn tất trích xuất {len(config.ALL_FEATURES)} đặc trưng (39 gốc + 11 cải tiến)!")
    print(f"  • Kích thước đầu ra : {len(df_final):,} dòng x {len(df_final.columns)} cột")
    print(f"  • Đã lưu file tại   : {config.FEATURES_DATA_PATH}")
    return df_final