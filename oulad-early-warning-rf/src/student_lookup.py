"""Tra cứu và demo dự báo theo từng sinh viên.
 
    from src.student_lookup import lookup_student, top_risk_students, demo
    lookup_student("FFF", "2014J", 123456)
    top_risk_students("FFF", "2014J", n=15)
    demo()
"""
import pandas as pd
 
from src import config
from src.feature_engineering import load_features
from src.main_model import split_data
from src.predictor import Predictor
 
# Các chỉ số hành vi dùng để so sánh sinh viên với nhóm tham chiếu
COMPARE_FEATURES = ["total_clicks", "active_days", "recency_days", "clicks_d21_28",
                    "activity_clicks_forumng", "activity_clicks_oucontent"]
 
_cache = {}
 
 
def _table():
    """Bảng đặc trưng + kết quả dự báo cho toàn bộ cohort (tính một lần)."""
    if "t" not in _cache:
        df = load_features()
        pred = Predictor().predict(df)
        t = pd.concat([df, pred], axis=1)
        t["tap_du_lieu"] = ["test" if x else "train" for x in split_data(df)]
        _cache["t"] = t
    return _cache["t"]
 
 
def _find(code_module, code_presentation, id_student):
    t = _table()
    m = t[(t["code_module"] == code_module) & (t["code_presentation"] == code_presentation) &
          (t["id_student"] == id_student)]
    if m.empty:
        raise KeyError(f"Không tìm thấy sinh viên {id_student} ({code_module}-{code_presentation}) trong cohort "
                       f"(có thể đã rút học phần trước ngày {config.CUTOFF_DAY} hoặc đăng ký muộn).")
    return t, m.iloc[0]
 
 
def lookup_student(code_module, code_presentation, id_student):
    """Dự báo cho 1 sinh viên và so sánh với trung vị nhóm không rút học phần cùng môn, cùng khóa."""
    t, row = _find(code_module, code_presentation, id_student)
    ref = t[(t["code_module"] == code_module) & (t["code_presentation"] == code_presentation) &
            (t[config.TARGET_COL] != config.POSITIVE_LABEL)]
    # thứ hạng nguy cơ trong cùng môn/khóa (1 = nguy cơ cao nhất)
    same = t[(t["code_module"] == code_module) & (t["code_presentation"] == code_presentation)]
    rank = int((same["risk_prob"] > row["risk_prob"]).sum() + 1)
    return {
        "id_student": int(id_student), "code_module": code_module, "code_presentation": code_presentation,
        "risk_prob": float(row["risk_prob"]), "risk_level": row["risk_level"],
        "predicted_label": row["predicted_label"], "thuc_te": row[config.TARGET_COL],
        "hang_nguy_co": f"{rank}/{len(same)}", "tap_du_lieu": row["tap_du_lieu"],
        "so_sanh_voi_nhom_khong_rut": {
            config.FEATURE_LABELS_VI[f]: (float(row[f]), float(ref[f].median())) for f in COMPARE_FEATURES},
    }
 
 
def print_student_report(code_module, code_presentation, id_student):
    r = lookup_student(code_module, code_presentation, id_student)
    print(f"Sinh viên {r['id_student']} | môn {r['code_module']} | khóa {r['code_presentation']}")
    print(f"  Nguy cơ rút học phần : {r['risk_prob']:.1%}  -> mức {r['risk_level']} "
          f"(hạng {r['hang_nguy_co']} trong môn/khóa)")
    print(f"  Dự báo / Thực tế     : {r['predicted_label']} / {r['thuc_te']}  [{r['tap_du_lieu']}]")
    print(f"  So với trung vị nhóm không rút (cùng môn, cùng khóa):")
    for k, (v, m) in r["so_sanh_voi_nhom_khong_rut"].items():
        print(f"    {k:<42s} sv = {v:8.1f} | nhóm = {m:8.1f}")
    if r["tap_du_lieu"] == "train":
        print("  (Lưu ý: sinh viên này thuộc tập train nên dự báo mang tính 'đã thấy dữ liệu'.)")
    return r
 
 
def top_risk_students(code_module, code_presentation, n=15, only_test=True):
    t = _table()
    s = t[(t["code_module"] == code_module) & (t["code_presentation"] == code_presentation)]
    if only_test:
        s = s[s["tap_du_lieu"] == "test"]
    cols = ["id_student", "risk_prob", "risk_level", "total_clicks", "active_days", "recency_days", config.TARGET_COL]
    return s.sort_values("risk_prob", ascending=False)[cols].head(n).reset_index(drop=True)
 
 
def demo(n=config.DEMO_N_STUDENTS, seed=1):
    """In bảng dự báo cho n sinh viên ngẫu nhiên ở tập test và 1 báo cáo chi tiết."""
    t = _table()
    s = t[t["tap_du_lieu"] == "test"].sample(n, random_state=seed)
    cols = config.PRIMARY_KEY + ["risk_prob", "risk_level", "total_clicks", "active_days", config.TARGET_COL]
    print(f"=== DEMO: {n} sinh viên ngẫu nhiên ở tập test (dự báo tại ngày {config.CUTOFF_DAY}) ===")
    print(s[cols].to_string(index=False))
    print("\n=== BÁO CÁO CHI TIẾT 1 SINH VIÊN ===")
    top = s.sort_values("risk_prob", ascending=False).iloc[0]
    print_student_report(top["code_module"], top["code_presentation"], int(top["id_student"]))
 
 
if __name__ == "__main__":
    demo()