import os
import pandas as pd
from src import config

def search_student_menu():
    """Tra cứu và in ra 39 đặc trưng của một id_student."""
    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
    if not os.path.exists(feat_path):
        print("\n⚠️ Lỗi: Chưa có file features_day28.csv! Vui lòng chạy Mục 1 (Tiền xử lý) trước.")
        return

    df = pd.read_csv(feat_path)
    # Lấy 5 mã id_student gợi ý
    sample_ids = df["id_student"].drop_duplicates().head(5).tolist()

    print("\n" + "="*65)
    print("🔍 TRA CỨU BỘ THUỘC TÍNH SINH VIÊN (OULAD)")
    print("="*65)
    print(f"💡 Gợi ý một số mã id_student có trong dữ liệu: {sample_ids}")
    
    raw_id = input("👉 Nhập mã sinh viên (id_student) cần tìm: ").strip()
    if not raw_id.isdigit():
        print("❌ Mã sinh viên phải là một số nguyên!")
        return

    student_id = int(raw_id)
    records = df[df["id_student"] == student_id]

    if records.empty:
        print(f"❌ Không tìm thấy id_student = {student_id} trong nhóm dự báo ngày {config.CUTOFF_DAY}.")
        return

    print(f"\n✅ Tìm thấy {len(records)} lượt đăng ký học phần của sinh viên [{student_id}]:")
    
    for idx, (_, row) in enumerate(records.iterrows(), 1):
        print("\n" + "-"*65)
        print(f"📌 LƯỢT ĐĂNG KÝ #{idx}: Học phần {row['code_module']} | Đợt mở lớp: {row.get('code_presentation', 'N/A')}")
        print(f"   Nhãn thực tế (Target): {row.get(config.TARGET_COL, 'N/A')}")
        print("-"*65)
        
        print("  [A] 7 Thuộc tính Nền & Nhân khẩu học:")
        for col in config.CATEGORICAL_FEATURES:
            print(f"      • {col:<26}: {row[col]}")
            
        print("\n  [B] 12 Thuộc tính Đăng ký & Tương tác tổng quát (Ngày 0-28):")
        for col in config.BASE_NUMERICAL_FEATURES:
            print(f"      • {col:<26}: {row[col]}")
            
        print("\n  [C] 20 Thuộc tính Lượt Click theo loại tài nguyên VLE:")
        for col in config.ACTIVITY_FEATURES:
            print(f"      • {col:<32}: {row[col]}")