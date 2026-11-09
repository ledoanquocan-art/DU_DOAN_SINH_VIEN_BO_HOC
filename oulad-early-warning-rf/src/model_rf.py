from sklearn.ensemble import RandomForestClassifier
from src import config
from src.pipeline_utils import get_user_param, train_and_save_model


def train_rf():
    """Mục 2.1: Hỏi từng tham số, huấn luyện Random Forest và lưu model kèm tham số."""
    print("\n" + "=" * 60)
    print("🌲 [2.1] NHẬP THAM SỐ HUẤN LUYỆN RANDOM FOREST")
    print("=" * 60)

    n_est = get_user_param(
        "n_estimators", 200, int,
        description="(Số lượng cây quyết định, VD: 100, 200)"
    )
    m_depth = get_user_param(
        "max_depth", 12, int,
        description="(Độ sâu tối đa của cây, gõ None nếu không giới hạn)"
    )
    m_leaf = get_user_param(
        "min_samples_leaf", 2, int,
        description="(Số mẫu tối thiểu tại nút lá, VD: 1, 2, 5)"
    )
    c_weight = get_user_param(
        "class_weight", "balanced_subsample", str,
        valid_options=["balanced", "balanced_subsample", None],
        description="(Trọng số lớp xử lý mất cân bằng, gõ None nếu tắt)"
    )

    params = {
        "n_estimators": n_est,
        "max_depth": m_depth,
        "min_samples_leaf": m_leaf,
        "class_weight": c_weight
    }

    estimator = RandomForestClassifier(
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=config.SEED,
        n_jobs=1,
        **params
    )

    return train_and_save_model(
        model_key="rf",
        model_name="Random Forest",
        estimator=estimator,
        params=params,
        scale_numerical=False
    )