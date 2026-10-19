from sklearn.svm import SVC
from src import config
from src.pipeline_utils import get_user_param, train_and_save_model

def train_svm():
    print("\n" + "=" * 60)
    print("📐 [2.3] NHẬP THAM SỐ HUẤN LUYỆN SVM")
    print("=" * 60)
    c_val = get_user_param("C", 10.0, float, description="(Hệ số phạt lỗi C, VD: 0.1, 1.0, 10.0)")
    c_weight = get_user_param(
        "class_weight", "balanced", str,
        valid_options=["balanced", None],
        description="(Trọng số lớp: balanced hoặc gõ None)"
    )
    kernel = get_user_param(
        "kernel", "rbf", str,
        valid_options=["rbf", "linear", "poly", "sigmoid"],
        description="(Hàm nhân Kernel)"
    )

    params = {
        "C": c_val,
        "class_weight": c_weight,
        "kernel": kernel
    }
    estimator = SVC(gamma="scale", probability=False, random_state=config.SEED, **params)
    return train_and_save_model("svm", "SVM", estimator, params, scale_numerical=True)