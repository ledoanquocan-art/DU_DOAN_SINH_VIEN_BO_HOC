"""Huấn luyện GaussianNB qua Pipeline và phép chia nhóm chung của dự án."""

import math

from sklearn.naive_bayes import GaussianNB

from src.pipeline_utils import get_user_param, train_and_save_model


def train_nb():
    """Điểm gọi từ main.py, mục 2 -> 4; trả về artifact của mô hình."""
    print("\n" + "=" * 65)
    print("[2.4] HUẤN LUYỆN GAUSSIAN NAIVE BAYES")
    print("=" * 65)

    while True:
        smoothing = get_user_param(
            "var_smoothing",
            1e-9,
            float,
            description="(Số dương hữu hạn; ví dụ 1e-11, 1e-9, 1e-7)",
        )
        # get_user_param hiện cho phép gõ None và float nhận cả nan/inf.
        if smoothing is not None and math.isfinite(smoothing) and smoothing > 0:
            break
        print("Giá trị không hợp lệ: var_smoothing phải hữu hạn và lớn hơn 0.")

    params = {"var_smoothing": smoothing}
    estimator = GaussianNB(**params)
    print("Tiền xử lý: One-Hot biến phân loại; chuẩn hóa các biến số.")
    print("Dùng CV 5 fold theo id_student trên train; chưa đánh giá test.")

    return train_and_save_model(
        model_key="nb",
        model_name="Gaussian Naive Bayes",
        estimator=estimator,
        params=params,
        scale_numerical=True,
    )
