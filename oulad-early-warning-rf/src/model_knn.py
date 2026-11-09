from sklearn.neighbors import KNeighborsClassifier
from src import config
from src.pipeline_utils import get_user_param, train_and_save_model


def train_knn():
    """
    Huấn luyện k-NN theo tham số nhập từ terminal, tích hợp đồng bộ với
    hệ thống Pipeline chung (5-Fold CV theo id_student, lưu vào results/models/).
    """
    print("\n" + "=" * 60)
    print("👥 [2.2] NHẬP THAM SỐ HUẤN LUYỆN k-NN")
    print("=" * 60)

    n_neighbors = get_user_param(
        "n_neighbors", 5, int,
        valid_options=[1, 3, 5, 7, 9, 11, 15, 21],
        description="(Số láng giềng k)"
    )
    weights = get_user_param(
        "weights", "uniform", str,
        valid_options=["uniform", "distance"],
        description="(Trọng số: uniform hoặc distance)"
    )
    metric = get_user_param(
        "metric", "euclidean", str,
        valid_options=["euclidean", "manhattan", "minkowski"],
        description="(Hàm khoảng cách)"
    )

    params = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
    }

    estimator = KNeighborsClassifier(**params)
    return train_and_save_model("knn", "k-NN", estimator, params, scale_numerical=True)
