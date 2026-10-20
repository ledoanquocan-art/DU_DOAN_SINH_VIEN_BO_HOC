import os
import glob
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

from src import config
from src.cohort_builder import build_cohort
from src.feature_engineering import generate_features
from src.pipeline_utils import (
    load_and_split_data,
    train_and_save_model,
    list_saved_models,
    open_file_on_os
)
from src.evaluator import plot_confusion_matrix
from src.predictor import create_sample_unlabeled_file

warnings.filterwarnings("ignore")

# =====================================================================
# CẤU HÌNH TRANG WEB & CSS GIAO DIỆN
# =====================================================================
st.set_page_config(
    page_title="OULAD Early Warning System - Nhóm 1 CMC",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4068 0%, #3b6978 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 1.75rem;
    }
    .main-header p {
        color: #e0ece4;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-left: 5px solid #3b6978;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎓 HỆ THỐNG CẢNH BÁO SỚM NGUY CƠ RÚT HỌC PHẦN (OULAD)</h1>
    <p>Dự báo tại thời điểm cuối ngày 28 (39 đặc trưng VLE & Nhân khẩu học) — Nhóm 1 Trường Đại học CMC</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR MENU ĐIỀU HƯỚNG (6 CHỨC NĂNG CHUẨN)
# =====================================================================
with st.sidebar:
    st.markdown("### 🧭 DANH MỤC CHỨC NĂNG")
    menu = st.radio(
        "Chọn chức năng thực hiện:",
        [
            "1. Chạy tiền xử lý dữ liệu",
            "2. Huấn luyện mô hình",
            "3. Chạy mô hình (Kiểm thử & Biểu đồ)",
            "4. Đối chứng đa mô hình",
            "5. Dự đoán dữ liệu không nhãn",
            "6. Tìm kiếm sinh viên (Tra cứu 39 cột)"
        ]
    )

    st.markdown("---")
    cohort_ready = os.path.exists(os.path.join(config.PROCESSED_DATA_DIR, "cohort_day28.csv"))
    feat_ready = os.path.exists(os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv"))
    saved_models = list_saved_models()
    eval_files = glob.glob(os.path.join(config.METRICS_DIR, "*_metrics.json"))

    st.markdown("### 📌 TRẠNG THÁI HỆ THỐNG")
    st.write(f"• Cohort ngày 28: {'✅ Sẵn sàng' if cohort_ready else '❌ Chưa tạo'}")
    st.write(f"• Bảng 39 đặc trưng: {'✅ Sẵn sàng' if feat_ready else '❌ Chưa tạo'}")
    st.write(f"• Mô hình đã train: **{len(saved_models)}** cấu hình")
    st.write(f"• Mô hình đã chạy Test: **{len(eval_files)}** cấu hình")

    st.markdown("---")
    st.caption("**Nhóm 1 - CMC University**\n"
               "- Lê Doãn Quốc An (Leader)\n"
               "- Trịnh Phan Tuấn Anh\n"
               "- Nông Đức Mạnh\n"
               "- Trần Ngọc Tuấn\n"
               "- Nguyễn Anh Đức")


# =====================================================================
# CHỨC NĂNG 1: TIỀN XỬ LÝ DỮ LIỆU
# =====================================================================
if menu == "1. Chạy tiền xử lý dữ liệu":
    st.subheader("🛠️ 1. Khảo sát OULAD, Lọc nhóm dự báo ngày 28 & Trích xuất 39 đặc trưng")
    st.write(
        "Quy trình tự động kiểm tra 7 bảng gốc trong `data/raw/`, lọc tuần tự 4 bước để thu về "
        "**27.421 lượt đăng ký hợp lệ** tại cuối ngày 28 và tổng hợp hành vi tương tác VLE từ ngày 0 đến ngày 28."
    )

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        run_prep = st.button("🚀 Bắt đầu chạy Tiền xử lý dữ liệu", type="primary", use_container_width=True)

    if run_prep:
        try:
            with st.spinner("⏳ [Bước 1/2] Đang lọc nhóm dự báo hợp lệ tại ngày 28 (build_cohort)..."):
                cohort_df = build_cohort()
            st.success(f"✔ Đã lưu `cohort_day28.csv` ({len(cohort_df):,} dòng).")

            with st.spinner("⏳ [Bước 2/2] Đang đọc studentVle.csv và trích xuất 39 đặc trưng (generate_features)..."):
                feat_df = generate_features()
            st.success(f"✔ Hoàn tất! Đã tạo `features_day28.csv` ({len(feat_df):,} dòng x {len(config.ALL_FEATURES)} đặc trưng).")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Lỗi khi chạy tiền xử lý: {e}")

    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path)
        n_total = len(df_feat)
        n_w = int((df_feat[config.TARGET_COL] == "Withdrawn").sum())
        n_nw = int((df_feat[config.TARGET_COL] == "Not_withdrawn").sum())
        n_stu = df_feat["id_student"].nunique()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng lượt đăng ký hợp lệ", f"{n_total:,}", f"{n_stu:,} sinh viên")
        m2.metric("Không rút (Not_withdrawn)", f"{n_nw:,}", f"{n_nw/n_total*100:.2f}%")
        m3.metric("Rút học phần (Withdrawn)", f"{n_w:,}", f"{n_w/n_total*100:.2f}%")
        m4.metric("Số đặc trưng đầu vào", f"{len(config.ALL_FEATURES)} cột", "7 phân loại + 32 biến số")

        c_left, c_right = st.columns([1, 1.4])
        with c_left:
            fig_dist = os.path.join(config.FIGURES_DIR, "fig3_1_label_distribution.png")
            if os.path.exists(fig_dist):
                st.image(fig_dist, caption="Hình 3.1: Phân phối nhãn trên 27.421 lượt đăng ký", use_container_width=True)
        with c_right:
            st.markdown("**Xem trước 10 dòng đầu của bảng `features_day28.csv`:**")
            st.dataframe(df_feat.head(10), use_container_width=True, height=300)
    else:
        st.info("💡 Chưa có dữ liệu đã xử lý. Hãy bấm nút **Bắt đầu chạy Tiền xử lý dữ liệu** ở trên.")


# =====================================================================
# CHỨC NĂNG 2: HUẤN LUYỆN MÔ HÌNH (NHẬP TỪNG THAM SỐ & LƯU MODEL ĐỘC LẬP)
# =====================================================================
elif menu == "2. Huấn luyện mô hình":
    st.subheader("⚙️ 2. Huấn luyện Mô hình với Tham số Tùy chỉnh")
    st.write(
        "Mỗi bộ tham số bạn chọn sẽ được huấn luyện trên tập Train, đánh giá chéo **5-Fold StratifiedGroupKFold** "
        "và lưu thành **một mô hình độc lập** kèm tên tham số trong `results/models/`."
    )

    if not feat_ready:
        st.warning("⚠️ Vui lòng chạy **Mục 1. Chạy tiền xử lý dữ liệu** trước khi huấn luyện!")
    else:
        algo = st.selectbox(
            "👉 Chọn thuật toán cần huấn luyện:",
            [
                "1. Random Forest (Mô hình chính)",
                "2. k-NN (K-Nearest Neighbors)",
                "3. SVM (Support Vector Machine)",
                "4. Bayes (Gaussian Naive Bayes)"
            ]
        )

        st.markdown("#### 🛠️ Thiết lập từng tham số đầu vào:")
        with st.form("train_model_form"):
            if algo.startswith("1."):
                c1, c2 = st.columns(2)
                with c1:
                    n_estimators = st.number_input("n_estimators (Số cây quyết định)", min_value=10, max_value=1000, value=200, step=10)
                    min_samples_leaf = st.number_input("min_samples_leaf (Số mẫu tối thiểu ở lá)", min_value=1, max_value=50, value=2, step=1)
                with c2:
                    max_depth_opt = st.selectbox("Chế độ max_depth", ["Giới hạn độ sâu (Nhập số)", "None (Không giới hạn)"])
                    max_depth_val = st.number_input("Giá trị max_depth (nếu giới hạn)", min_value=2, max_value=100, value=12, step=1)
                    class_weight_opt = st.selectbox("class_weight (Trọng số lớp)", ["balanced_subsample", "balanced", "None"])

            elif algo.startswith("2."):
                c1, c2, c3 = st.columns(3)
                with c1:
                    n_neighbors = st.number_input("n_neighbors (Số láng giềng k)", min_value=1, max_value=101, value=5, step=2)
                with c2:
                    weights = st.selectbox("weights (Trọng số khoảng cách)", ["uniform", "distance"])
                with c3:
                    p_metric = st.selectbox("p (Khoảng cách: 1=Manhattan, 2=Euclid)", [2, 1])

            elif algo.startswith("3."):
                c1, c2, c3 = st.columns(3)
                with c1:
                    c_val = st.number_input("C (Hệ số phạt lỗi)", min_value=0.01, max_value=100.0, value=10.0, step=0.5, format="%.2f")
                with c2:
                    svm_cw = st.selectbox("class_weight (Trọng số lớp)", ["balanced", "None"])
                with c3:
                    kernel = st.selectbox("kernel (Hàm nhân - Khuyên dùng rbf)", ["rbf", "poly", "sigmoid", "linear"])

            elif algo.startswith("4."):
                var_smoothing = st.select_slider(
                    "var_smoothing (Hệ số làm trơn phương sai)",
                    options=[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    value=1e-9
                )

            submit_train = st.form_submit_button("🚀 Huấn luyện & Lưu mô hình với bộ tham số này", type="primary")

        if submit_train:
            try:
                with st.spinner("⏳ Đang chạy 5-Fold Cross-Validation và huấn luyện mô hình..."):
                    if algo.startswith("1."):
                        m_depth = None if max_depth_opt.startswith("None") else int(max_depth_val)
                        c_weight = None if class_weight_opt == "None" else class_weight_opt
                        params = {
                            "n_estimators": int(n_estimators),
                            "max_depth": m_depth,
                            "min_samples_leaf": int(min_samples_leaf),
                            "class_weight": c_weight
                        }
                        est = RandomForestClassifier(
                            criterion="gini", max_features="sqrt", bootstrap=True,
                            random_state=config.SEED, n_jobs=1, **params
                        )
                        artifact = train_and_save_model("rf", "Random Forest", est, params, scale_numerical=False)

                    elif algo.startswith("2."):
                        params = {"n_neighbors": int(n_neighbors), "weights": weights, "p": int(p_metric)}
                        est = KNeighborsClassifier(**params)
                        artifact = train_and_save_model("knn", "k-NN", est, params, scale_numerical=True)

                    elif algo.startswith("3."):
                        c_weight = None if svm_cw == "None" else svm_cw
                        params = {"C": float(c_val), "class_weight": c_weight, "kernel": kernel}
                        est = SVC(gamma="scale", probability=False, random_state=config.SEED, **params)
                        artifact = train_and_save_model("svm", "SVM", est, params, scale_numerical=True)

                    elif algo.startswith("4."):
                        params = {"var_smoothing": float(var_smoothing)}
                        est = GaussianNB(priors=None, **params)
                        artifact = train_and_save_model("nb", "GaussianNB", est, params, scale_numerical=True)

                st.success(
                    f"✅ Đã huấn luyện & lưu thành công: **{artifact['model_label']}**\n\n"
                    f"• **CV Accuracy**: `{artifact['cv_accuracy']}%` | "
                    f"**CV Macro-F1**: `{artifact['cv_macro_f1']}% ± {artifact['cv_macro_f1_std']}%`\n\n"
                    f"• **File lưu**: `results/models/{artifact['model_id']}.joblib`"
                )
            except Exception as e:
                st.error(f"❌ Lỗi khi huấn luyện: {e}")

        # Hiển thị danh sách các mô hình đã lưu
        models_now = list_saved_models()
        if models_now:
            st.markdown("---")
            st.markdown("#### 📦 Danh sách tất cả các mô hình (cấu hình tham số) đã huấn luyện:")
            summary_rows = []
            for m in models_now:
                summary_rows.append({
                    "Mã File (model_id)": m["model_id"],
                    "Tên Mô hình & Tham số": m["model_label"],
                    "CV Accuracy (%)": m["cv_accuracy"],
                    "CV Macro-F1 (%)": f"{m['cv_macro_f1']} ± {m['cv_macro_f1_std']}",
                    "Trạng thái Test": "✅ Đã chạy Test" if m.get("test_metrics") else "🆕 Chưa chạy Test",
                    "Thời gian tạo": m.get("trained_at", "")
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# =====================================================================
# CHỨC NĂNG 3: CHẠY MÔ HÌNH (XUẤT KẾT QUẢ, BIỂU ĐỒ & LƯU FILE)
# =====================================================================
elif menu == "3. Chạy mô hình (Kiểm thử & Biểu đồ)":
    st.subheader("🚀 3. Chạy Kiểm thử Mô hình trên Tập Test (5.465 mẫu)")

    models = list_saved_models()
    if not models:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện! Vui lòng vào **Mục 2. Huấn luyện mô hình** trước.")
    else:
        model_options = {
            f"[{i+1}] {m['model_label']} (CV Macro-F1: {m['cv_macro_f1']}%)": m
            for i, m in enumerate(models)
        }
        chosen_label = st.selectbox("👉 Chọn mô hình đã huấn luyện để chạy:", list(model_options.keys()))
        selected = model_options[chosen_label]

        if st.button("▶️ Chạy Mô hình này & Xuất biểu đồ", type="primary"):
            with st.spinner(f"⏳ Đang chạy kiểm thử {selected['model_label']} trên tập Test..."):
                pipe = selected["pipeline"]
                model_id = selected["model_id"]
                model_label = selected["model_label"]

                _, _, _, X_test, y_test, _, _ = load_and_split_data()
                y_pred = pipe.predict(X_test)

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["Not_withdrawn", "Withdrawn"]).ravel()
                metrics = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
                    "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")) * 100, 2),
                    "precision_w": round(float(precision_score(y_test, y_pred, pos_label="Withdrawn", zero_division=0)) * 100, 2),
                    "recall_w": round(float(recall_score(y_test, y_pred, pos_label="Withdrawn", zero_division=0)) * 100, 2),
                    "f1_w": round(float(f1_score(y_test, y_pred, pos_label="Withdrawn", zero_division=0)) * 100, 2),
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
                }

                if hasattr(pipe.named_steps["model"], "predict_proba"):
                    classes = list(pipe.named_steps["model"].classes_)
                    w_idx = classes.index("Withdrawn")
                    y_prob = pipe.predict_proba(X_test)[:, w_idx]
                    y_true_bin = (y_test == "Withdrawn").astype(int)
                    metrics["ap_withdrawn"] = round(float(average_precision_score(y_true_bin, y_prob)), 4)
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true_bin, y_prob)), 4)

                # Vẽ Ma trận nhầm lẫn
                cm_fig_path = os.path.join(config.FIGURES_DIR, f"cm_{model_id}.png")
                plot_confusion_matrix(tn, fp, fn, tp, model_label, cm_fig_path)

                # Nếu là Random Forest thì vẽ thêm Permutation Importance
                pi_path = os.path.join(config.FIGURES_DIR, f"perm_importance_{model_id}.png")
                if selected["model_key"] == "rf":
                    perm = permutation_importance(
                        pipe, X_test, y_test, scoring="f1_macro",
                        n_repeats=5, random_state=config.SEED, n_jobs=2
                    )
                    sorted_idx = perm.importances_mean.argsort()[-8:]
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.barh(
                        [config.ALL_FEATURES[i] for i in sorted_idx],
                        perm.importances_mean[sorted_idx] * 100,
                        xerr=perm.importances_std[sorted_idx] * 100,
                        color="#3b6978"
                    )
                    ax.set_xlabel("Mức giảm Macro-F1 khi hoán vị (điểm phần trăm)")
                    ax.set_title(f"Top 8 Đặc trưng quan trọng nhất\n{model_label}", fontsize=10)
                    fig.tight_layout()
                    plt.savefig(pi_path, dpi=200)
                    plt.close(fig)

                # Cập nhật file .joblib và xuất JSON
                selected_copy = dict(selected)
                filepath = selected_copy.pop("filepath")
                selected_copy["test_metrics"] = metrics
                joblib.dump(selected_copy, filepath)

                json_payload = {
                    "model_id": model_id,
                    "model_key": selected["model_key"],
                    "model_name": selected["model_name"],
                    "model_label": model_label,
                    "params": selected["params"],
                    "cv_accuracy": selected["cv_accuracy"],
                    "cv_macro_f1": selected["cv_macro_f1"],
                    "test_metrics": metrics
                }
                json_path = os.path.join(config.METRICS_DIR, f"{model_id}_metrics.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_payload, f, indent=4, ensure_ascii=False)

                selected["test_metrics"] = metrics
                st.success(f"✅ Đã chạy xong và lưu kết quả tại `results/metrics/{model_id}_metrics.json`!")

        # Hiển thị kết quả nếu mô hình đã có test_metrics
        if selected.get("test_metrics"):
            tm = selected["test_metrics"]
            model_id = selected["model_id"]
            st.markdown(f"#### 📊 Kết quả đánh giá trên Tập Test: `{selected['model_label']}`")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Accuracy", f"{tm['accuracy']}%")
            k2.metric("Macro-F1", f"{tm['macro_f1']}%")
            k3.metric("Precision (Withdrawn)", f"{tm['precision_w']}%")
            k4.metric("Recall (Withdrawn)", f"{tm['recall_w']}%")
            k5.metric("F1-Score (Withdrawn)", f"{tm['f1_w']}%")

            d1, d2, d3 = st.columns(3)
            d1.info(f"🎯 **Phát hiện đúng Rút học phần (TP):** `{tm['tp']:,}` ca")
            d2.warning(f"⚠️ **Bỏ sót Rút học phần (FN):** `{tm['fn']:,}` ca")
            d3.error(f"🔔 **Cảnh báo nhầm (FP):** `{tm['fp']:,}` ca")

            fig_col1, fig_col2 = st.columns(2)
            cm_path = os.path.join(config.FIGURES_DIR, f"cm_{model_id}.png")
            pi_path = os.path.join(config.FIGURES_DIR, f"perm_importance_{model_id}.png")

            with fig_col1:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Biểu đồ Ma trận nhầm lẫn (Confusion Matrix)", use_container_width=True)
            with fig_col2:
                if os.path.exists(pi_path):
                    st.image(pi_path, caption="Biểu đồ Top 8 Đặc trưng quan trọng nhất (Permutation Importance)", use_container_width=True)


# =====================================================================
# CHỨC NĂNG 4: ĐỐI CHỨNG ĐA MÔ HÌNH (CHỌN >= 2 MÔ HÌNH ĐÃ CHẠY)
# =====================================================================
elif menu == "4. Đối chứng đa mô hình":
    st.subheader("⚖️ 4. Đối chứng các Mô hình Đã Chạy Kiểm thử")

    metric_files = sorted(glob.glob(os.path.join(config.METRICS_DIR, "*_metrics.json")))
    if len(metric_files) < 2:
        st.warning(
            f"⚠️ Hiện mới có **{len(metric_files)}** mô hình được chạy ở Mục 3. "
            "Vui lòng vào **Mục 3** chạy ít nhất 2 mô hình trước khi thực hiện Đối chứng!"
        )
    else:
        records = []
        for f in metric_files:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if "model_label" not in data:
                    p_str = ", ".join([f"{k}={v}" for k, v in data.get("params", {}).items()])
                    data["model_label"] = f"{data.get('model_name', 'Model')} [{p_str}]"
                records.append(data)

        label_map = {r["model_label"]: r for r in records}
        selected_labels = st.multiselect(
            "👉 Chọn từ 2 mô hình trở lên để xuất báo cáo đối chứng:",
            options=list(label_map.keys()),
            default=list(label_map.keys())
        )

        if len(selected_labels) < 2:
            st.error("❌ Vui lòng chọn ít nhất **2 mô hình** để so sánh!")
        else:
            chosen_records = [label_map[lbl] for lbl in selected_labels]
            rows = []
            for r in chosen_records:
                m = r["test_metrics"]
                short_params = ", ".join([f"{k}={v}" for k, v in r.get("params", {}).items()])
                rows.append({
                    "Mô hình & Tham số": r["model_label"],
                    "CV Macro-F1 (%)": r.get("cv_macro_f1", "N/A"),
                    "Test Accuracy (%)": m["accuracy"],
                    "Test Macro-F1 (%)": m["macro_f1"],
                    "Precision W (%)": m["precision_w"],
                    "Recall W (%)": m["recall_w"],
                    "F1 W (%)": m["f1_w"],
                    "Đúng W (TP)": m["tp"],
                    "Bỏ sót W (FN)": m["fn"],
                    "Cảnh báo nhầm (FP)": m["fp"],
                    "_plot_label": f"{r['model_name']}\n({short_params})"
                })

            df_cmp = pd.DataFrame(rows)
            display_cols = [c for c in df_cmp.columns if c != "_plot_label"]

            st.markdown("#### 📋 Bảng báo cáo đối chứng các mô hình đã chọn:")
            st.dataframe(df_cmp[display_cols], use_container_width=True)

            csv_path = os.path.join(config.METRICS_DIR, "comparison_report.csv")
            df_cmp[display_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

            # Vẽ biểu đồ đối chứng
            x = np.arange(len(df_cmp))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(8.5, len(df_cmp) * 2.3), 4.8))
            b1 = ax.bar(x - width / 2, df_cmp["Test Accuracy (%)"], width, label="Accuracy (%)", color="#3b6978")
            b2 = ax.bar(x + width / 2, df_cmp["Test Macro-F1 (%)"], width, label="Macro-F1 (%)", color="#98b4be")

            ax.set_ylabel("Tỷ lệ (%)")
            ax.set_ylim(0, 108)
            ax.set_xticks(x, labels=df_cmp["_plot_label"], fontsize=9)
            ax.set_title("Đối chứng Accuracy và Macro-F1 trên Tập Test giữa các mô hình đã chọn")
            ax.legend()
            ax.bar_label(b1, padding=3, fmt="%.1f", fontsize=9)
            ax.bar_label(b2, padding=3, fmt="%.1f", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()

            fig_path = os.path.join(config.FIGURES_DIR, "fig3_3_model_comparison.png")
            plt.savefig(fig_path, dpi=200)
            plt.close(fig)

            st.image(fig_path, caption="Hình 3.3: Biểu đồ đối chứng Accuracy và Macro-F1", use_container_width=True)
            st.download_button(
                "📥 Tải bảng báo cáo đối chứng (CSV)",
                data=df_cmp[display_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="comparison_report.csv",
                mime="text/csv"
            )


# =====================================================================
# CHỨC NĂNG 5: DỰ ĐOÁN FILE DỮ LIỆU KHÔNG NHÃN & MỞ FILE
# =====================================================================
elif menu == "5. Dự đoán dữ liệu không nhãn":
    st.subheader("🔮 5. Dự đoán Dữ liệu Không Nhãn bằng Mô hình Đã Huấn luyện")

    models = list_saved_models()
    if not models:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện! Vui lòng vào **Mục 2** để huấn luyện trước.")
    else:
        # Tạo danh sách hiển thị mô hình kèm độ chính xác
        model_choices = {}
        for i, m in enumerate(models, 1):
            lbl = m.get("model_label", m["model_name"])
            if m.get("test_metrics"):
                tm = m["test_metrics"]
                acc_txt = f"Test Acc: {tm['accuracy']}% | Test Macro-F1: {tm['macro_f1']}% | Recall W: {tm['recall_w']}%"
            else:
                acc_txt = f"CV Acc: {m['cv_accuracy']}% | CV Macro-F1: {m['cv_macro_f1']}%"
            model_choices[f"[{i}] {lbl}  —  🎯 {acc_txt}"] = m

        chosen_model_key = st.selectbox("👉 Bước 1: Chọn mô hình muốn sử dụng (kèm độ chính xác):", list(model_choices.keys()))
        chosen_model = model_choices[chosen_model_key]

        st.markdown("#### 📂 Bước 2: Cung cấp file dữ liệu không nhãn (`.csv`)")
        data_source = st.radio(
            "Chọn nguồn dữ liệu đầu vào:",
            ["Dùng file mẫu không nhãn có sẵn (50 sinh viên kèm ca mẫu dòng 334)", "Tải lên file CSV từ máy tính"],
            horizontal=True
        )

        df_input = None
        source_name = "sample_unlabeled_students"

        if data_source.startswith("Dùng file mẫu"):
            sample_path = create_sample_unlabeled_file()
            df_input = pd.read_csv(sample_path)
            st.caption(f"Đang sử dụng file mẫu: `{sample_path}` ({len(df_input)} bản ghi)")
        else:
            uploaded = st.file_uploader("Tải lên file CSV chứa 39 cột đặc trưng:", type=["csv"])
            if uploaded is not None:
                df_input = pd.read_csv(uploaded)
                source_name = os.path.splitext(uploaded.name)[0]

        auto_open = st.checkbox("📂 Tự động mở luôn file kết quả trên màn hình máy tính (Excel/CSV) sau khi dự đoán", value=True)

        if df_input is not None:
            missing_cols = [c for c in config.ALL_FEATURES if c not in df_input.columns]
            if missing_cols:
                st.error(f"❌ File dữ liệu thiếu các cột đặc trưng bắt buộc: {missing_cols}")
            else:
                if st.button("🚀 Chạy Dự đoán & Xuất file có nhãn", type="primary"):
                    pipe = chosen_model["pipeline"]
                    model_id = chosen_model["model_id"]

                    X_unlabeled = df_input[config.ALL_FEATURES]
                    df_out = df_input.copy()
                    df_out["predicted_target"] = pipe.predict(X_unlabeled)

                    try:
                        classes = list(pipe.named_steps["model"].classes_)
                        w_idx = classes.index("Withdrawn")
                        probs = pipe.predict_proba(X_unlabeled)[:, w_idx] * 100
                        df_out["withdrawn_probability_%"] = probs.round(2)
                    except Exception:
                        df_out["withdrawn_probability_%"] = "N/A (SVM probability=False)"

                    out_filename = f"{source_name}__predicted_by__{model_id}.csv"
                    out_path = os.path.join(config.PREDICTIONS_DIR, out_filename)
                    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

                    n_total = len(df_out)
                    n_w = int((df_out["predicted_target"] == "Withdrawn").sum())
                    n_nw = n_total - n_w

                    st.success(f"✅ Đã gán nhãn xong {n_total} dòng và lưu tại: `data/predictions/{out_filename}`")

                    p1, p2, p3 = st.columns(3)
                    p1.metric("Tổng số dòng dự đoán", f"{n_total:,}")
                    p2.metric("Dự đoán Rút học phần (Withdrawn)", f"{n_w:,} ca", f"{n_w/n_total*100:.2f}%")
                    p3.metric("Dự đoán Không rút (Not_withdrawn)", f"{n_nw:,} ca", f"{n_nw/n_total*100:.2f}%")

                    st.markdown("#### 📋 Bảng dữ liệu sau khi đã gán nhãn dự đoán:")
                    priority_cols = [
                        c for c in ["code_module", "code_presentation", "id_student", "predicted_target", "withdrawn_probability_%"]
                        if c in df_out.columns
                    ]
                    other_cols = [c for c in df_out.columns if c not in priority_cols]
                    st.dataframe(df_out[priority_cols + other_cols], use_container_width=True)

                    st.download_button(
                        "📥 Tải file CSV đã có nhãn về máy",
                        data=df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                        file_name=out_filename,
                        mime="text/csv"
                    )

                    if auto_open:
                        open_file_on_os(out_path)
                        st.info(f"📂 Đã kích hoạt mở trực tiếp file `{out_path}` trên hệ điều hành!")


# =====================================================================
# CHỨC NĂNG 6: TÌM KIẾM SINH VIÊN (TRA CỨU BỘ 39 THUỘC TÍNH)
# =====================================================================
elif menu == "6. Tìm kiếm sinh viên (Tra cứu 39 cột)":
    st.subheader("🔍 6. Tra cứu Hồ sơ & 39 Thuộc tính của Sinh viên")

    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
    if not os.path.exists(feat_path):
        st.warning("⚠️ Chưa có file `features_day28.csv`! Vui lòng chạy **Mục 1. Chạy tiền xử lý dữ liệu** trước.")
    else:
        df = pd.read_csv(feat_path)
        sample_ids = df["id_student"].drop_duplicates().head(8).tolist()
        st.caption(f"💡 Gợi ý một số mã `id_student` có trong dữ liệu: `{sample_ids}` (hoặc sinh viên dòng 334: `{int(df.iloc[334]['id_student'])}`)")

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            student_id_input = st.text_input("👉 Nhập mã sinh viên (id_student) cần tra cứu:", value=str(int(df.iloc[334]["id_student"])))

        if student_id_input.strip().isdigit():
            sid = int(student_id_input.strip())
            records = df[df["id_student"] == sid]

            if records.empty:
                st.error(f"❌ Không tìm thấy sinh viên có mã `id_student = {sid}` trong nhóm dự báo ngày 28.")
            else:
                st.success(f"✅ Tìm thấy **{len(records)}** lượt đăng ký học phần của sinh viên `id_student = {sid}`")

                for idx, (_, row) in enumerate(records.iterrows(), 1):
                    target_val = row.get(config.TARGET_COL, "N/A")
                    badge = "🔴 Withdrawn (Rút học phần)" if target_val == "Withdrawn" else "🟢 Not_withdrawn (Không rút)"
                    with st.expander(
                        f"📌 Lượt đăng ký #{idx}: Học phần {row['code_module']} | Đợt mở lớp {row.get('code_presentation', 'N/A')} — Nhãn thực tế: {badge}",
                        expanded=True
                    ):
                        t1, t2, t3 = st.columns(3)
                        with t1:
                            st.markdown("**[A] 7 Thuộc tính Nền & Nhân khẩu học**")
                            df_cat = pd.DataFrame({
                                "Thuộc tính": config.CATEGORICAL_FEATURES,
                                "Giá trị": [str(row[c]) for c in config.CATEGORICAL_FEATURES]
                            })
                            st.dataframe(df_cat, use_container_width=True, hide_index=True)

                        with t2:
                            st.markdown("**[B] 12 Thuộc tính Đăng ký & Tương tác (Ngày 0–28)**")
                            df_num = pd.DataFrame({
                                "Thuộc tính": config.BASE_NUMERICAL_FEATURES,
                                "Giá trị": [row[c] for c in config.BASE_NUMERICAL_FEATURES]
                            })
                            st.dataframe(df_num, use_container_width=True, hide_index=True)

                        with t3:
                            st.markdown("**[C] 20 Thuộc tính Click theo Tài nguyên VLE**")
                            df_act = pd.DataFrame({
                                "Loại tài nguyên VLE": config.ACTIVITY_FEATURES,
                                "Số Click": [int(row[c]) for c in config.ACTIVITY_FEATURES]
                            })
                            st.dataframe(df_act, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Vui lòng nhập mã `id_student` là một số nguyên!")