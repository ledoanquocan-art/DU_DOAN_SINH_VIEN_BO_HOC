"""Predictor: nạp Random Forest đã huấn luyện và dự báo nguy cơ rút học phần."""
import joblib
import pandas as pd
 
from src import config
 
 
def risk_level(prob):
    """Phân mức rủi ro theo ngưỡng trong config (Thấp / Trung bình / Cao)."""
    low, mid, high = config.RISK_LABELS
    if prob >= config.RISK_HIGH_MIN:
        return high
    return mid if prob >= config.RISK_LOW_MAX else low
 
 
class Predictor:
    def __init__(self, model_path=config.RF_MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle.get("threshold", config.DECISION_THRESHOLD)
 
    def predict(self, df):
        """df: bảng có đủ các cột ALL_FEATURES (có thể kèm PRIMARY_KEY). Giữ nguyên index."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu các cột đặc trưng: {missing}")
        prob = self.model.predict_proba(df[self.features])[:, 1]
        out = pd.DataFrame({
            "risk_prob": prob.round(4),
            "risk_level": [risk_level(p) for p in prob],
            "predicted_label": [config.POSITIVE_LABEL if p >= self.threshold else "Not_" + config.POSITIVE_LABEL.lower()
                                for p in prob],
        }, index=df.index)
        return out

 