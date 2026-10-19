"""Đối chứng các mô hình đã qua Mục 3 trên cùng tập test hiện tại.

Không fit lại mô hình, không thay đổi file .joblib và không chọn tham số theo test.
CV lấy từ artifact; các độ đo test được tính lại để tránh ghép kết quả JSON cũ.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.pipeline_utils import list_saved_models, load_and_split_data

LABELS = ["Not_withdrawn", "Withdrawn"]
POSITIVE = "Withdrawn"


def _choose_models(models):
    """Đọc số thứ tự, loại số lặp, giữ thứ tự chọn và cho phép hủy."""
    while True:
        raw = input("\nChọn ít nhất 2 mô hình, ví dụ 1,2,3 (0 để quay lại): ").strip()
        if raw == "0":
            return []
        tokens = raw.replace(",", " ").split()
        try:
            indexes = list(dict.fromkeys(int(token) for token in tokens))
        except ValueError:
            print("Chỉ nhập các số thứ tự, ngăn cách bằng dấu phẩy hoặc khoảng trắng.")
            continue
        if len(indexes) < 2:
            print("Cần ít nhất 2 mô hình khác nhau.")
            continue
        if any(index < 1 or index > len(models) for index in indexes):
            print(f"Số thứ tự phải trong khoảng 1 đến {len(models)}.")
            continue
        selected = [models[index - 1] for index in indexes]
        ids = [model["model_id"] for model in selected]
        if len(set(ids)) != len(ids):
            print("Các artifact được chọn trùng model_id; kiểm tra thư mục mô hình.")
            continue
        return selected


def _positive_scores(pipe, X):
    """Trả điểm xếp hạng lớp Withdrawn, không coi margin SVM là xác suất."""
    classes = list(pipe.classes_)
    if len(classes) != 2 or set(classes) != set(LABELS):
        raise ValueError(f"Nhãn mô hình không đúng bài toán nhị phân: {classes}")
    positive_index = classes.index(POSITIVE)

    if hasattr(pipe, "predict_proba"):
        scores = np.asarray(pipe.predict_proba(X))[:, positive_index]
        source = "predict_proba"
    elif hasattr(pipe, "decision_function"):
        scores = np.asarray(pipe.decision_function(X))
        if scores.ndim != 1:
            raise ValueError("decision_function phải trả một điểm cho mỗi mẫu.")
        # Với bài toán nhị phân, margin dương ứng với classes_[1].
        if positive_index == 0:
            scores = -scores
        source = "decision_function"
    else:
        return None, "unavailable"

    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(X),) or not np.isfinite(scores).all():
        raise ValueError("Điểm dự báo sai kích thước hoặc chứa NaN/Infinity.")
    return scores, source


def _evaluate_model(artifact, X_test, y_test):
    """Đơn vị % cho Accuracy/F1/Precision/Recall; AP và ROC-AUC từ 0 đến 1."""
    pipe = artifact["pipeline"]
    pred = np.asarray(pipe.predict(X_test))
    if pred.shape != (len(y_test),) or not set(pred).issubset(LABELS):
        raise ValueError("Nhãn dự đoán sai kích thước hoặc không đúng LABELS.")
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=LABELS).ravel()
    scores, score_source = _positive_scores(pipe, X_test)
    binary_target = np.asarray(y_test) == POSITIVE
    row = {
        "model_id": artifact["model_id"],
