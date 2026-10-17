import sys
from src.cohort_builder import build_cohort

# Tạm thời bọc trong try-except để main.py không bị crash khi các bạn khác chưa làm xong file
try:
    from src.feature_engineering import generate_features
    from src.model_rf import train_rf
    from src.model_knn import train_knn
    from src.model_svm import train_svm
    from src.model_nb import train_nb
    from src.evaluator import run_saved_model_menu
    from src.compare_models import compare_selected_models_menu
    from src.predictor import predict_unlabeled_file_menu
except ImportError:
    pass

from src.student_lookup import search_student_menu

def train_submenu():
    while True:
        print("\n" + "-"*50)
        print("🛠️  2. HUẤN LUYỆN MÔ HÌNH (NHẬP THAM SỐ & LƯU MODEL)")
        print("-"*50)
        print("   1. Random Forest")
        print("   2. k-NN")
        print("   3. SVM")
        print("   4. Bayes (Gaussian Naive Bayes)")
        print("   0. Quay lại Menu chính")

        sub_choice = input("\n👉 Chọn mô hình cần huấn luyện (0-4): ").strip()
        try:
            if sub_choice == "1":
                train_rf()
            elif sub_choice == "2":
                train_knn()
            elif sub_choice == "3":
                train_svm()
            elif sub_choice == "4":
                train_nb()
            elif sub_choice == "0":
                break
            else:
                print("❌ Vui lòng chọn từ 0 đến 4!")
        except NameError:
            print("⚠️ Chức năng này chưa hoàn thiện (Thành viên chưa nộp code).")
        except Exception as e:
            print(f"⚠️ Lỗi khi huấn luyện: {e}")

def main():
    while True:
        print("\n" + "="*65)
        print("🎓 HỆ THỐNG CẢNH BÁO NGUY CƠ RÚT HỌC PHẦN OULAD - NHÓM 1 CMC")
        print("="*65)
        print("  1. Chạy tiền xử lý dữ liệu")
        print("  2. Huấn luyện mô hình")
        print("       1. Random Forest | 2. k-NN | 3. SVM | 4. Bayes")
        print("  3. Chạy mô hình (Chọn mô hình đã train -> Xuất biểu đồ & lưu)")
        print("  4. Đối chứng (Chọn >= 2 mô hình đã chạy -> Xuất báo cáo)")
        print("  5. Dự đoán (Nhập file không nhãn -> Dự đoán & Mở file)")
        print("  6. Tìm SV (Tra cứu 39 thuộc tính theo mã sinh viên)")
        print("  0. Thoát")
        print("="*65)

        choice = input("👉 Nhập lựa chọn của bạn (0-6): ").strip()

        try:
            if choice == "1":
                print("\n⏳ [Bước 1/2] Đang lọc nhóm dự báo hợp lệ tại ngày 28...")
                build_cohort()
                print("⏳ [Bước 2/2] Đang tổng hợp 39 đặc trưng VLE (ngày 0-28)...")
                try:
                    generate_features()
                    print("✅ Hoàn thành tiền xử lý dữ liệu!")
                except NameError:
                    print("⚠️ Tuấn Anh chưa hoàn thành hàm generate_features().")
            elif choice == "2":
                train_submenu()
            elif choice == "3":
                try:
                    run_saved_model_menu()
                except NameError:
                    print("⚠️ Ngọc Tuấn chưa hoàn thành hàm run_saved_model_menu().")
            elif choice == "4":
                try:
                    compare_selected_models_menu()
                except NameError:
                    print("⚠️ Anh Đức chưa hoàn thành hàm compare_selected_models_menu().")
            elif choice == "5":
                try:
                    predict_unlabeled_file_menu()
                except NameError:
                    print("⚠️ Đức Mạnh chưa hoàn thành hàm predict_unlabeled_file_menu().")
            elif choice == "6":
                search_student_menu()
            elif choice == "0":
                print("👋 Đã thoát chương trình. Chúc nhóm bảo vệ thành công!")
                sys.exit(0)
            else:
                print("❌ Lựa chọn không hợp lệ, vui lòng nhập từ 0 đến 6!")
        except Exception as e:
            print(f"\n⚠️ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()