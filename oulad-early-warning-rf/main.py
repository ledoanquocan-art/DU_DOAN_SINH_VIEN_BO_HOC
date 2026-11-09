import sys
import warnings
warnings.filterwarnings("ignore")

from src import config
from src.cohort_builder import build_cohort
from src.student_lookup import search_student_menu


def train_submenu():
    while True:
        print("\n" + "-" * 58)
        print("🛠️  2. HUẤN LUYỆN MÔ HÌNH (NHẬP THAM SỐ & HIỂN THỊ % TIẾN TRÌNH)")
        print("-" * 58)
        print("   1. Random Forest")
        print("   2. k-NN")
        print("   3. SVM")
        print("   4. Bayes (Gaussian Naive Bayes)")
        print("   0. Quay lại Menu chính")

        sub_choice = input("\n👉 Chọn mô hình cần huấn luyện (0-4): ").strip()
        try:
            if sub_choice == "1":
                from src.model_rf import train_rf
                train_rf()
            elif sub_choice == "2":
                from src.model_knn import train_knn
                train_knn()
            elif sub_choice == "3":
                from src.model_svm import train_svm
                train_svm()
            elif sub_choice == "4":
                from src.model_nb import train_nb
                train_nb()
            elif sub_choice == "0":
                break
            else:
                print("❌ Vui lòng chọn từ 0 đến 4!")
        except ImportError as e:
            print(f"⚠️ Mô hình này chưa có hàm huấn luyện hoặc file còn trống ({e}).")
        except Exception as e:
            print(f"⚠️ Lỗi khi huấn luyện: {e}")


def main():
    while True:
        print("\n" + "=" * 72)
        print("🎓 HỆ THỐNG CẢNH BÁO NGUY CƠ BỎ HỌC DỰA TRÊN DỮ LIỆU VLE (OU-LAD)")
        print("=" * 72)
        print("  1. Chạy tiền xử lý dữ liệu (Lọc Cohort, Tạo 39 đặc trưng & Chia Train/Test)")
        print("  2. Huấn luyện mô hình (Có thanh % tiến trình & báo đường dẫn Train/Test)")
        print("       1. Random Forest | 2. k-NN | 3. SVM | 4. Bayes")
        print("  3. Chạy mô hình (Có thanh % tiến trình -> Xuất kết quả, biểu đồ & lưu)")
        print("  4. Đối chứng (Chọn >= 2 mô hình đã chạy -> Xuất báo cáo so sánh)")
        print("  5. Dự đoán (Nhập file không nhãn -> Chọn mô hình -> Xuất & mở file)")
        print("  6. Tìm SV (Nhập mã SV -> Trả ra bộ 39 thuộc tính của SV đó)")
        print("  0. Thoát")
        print("=" * 72)

        choice = input("👉 Nhập lựa chọn của bạn (0-6): ").strip()

        try:
            if choice == "1":
                print("\n⏳ [Bước 1/3] Đang lọc nhóm dự báo hợp lệ tại ngày 28...")
                build_cohort()
                print("\n⏳ [Bước 2/3] Đang tổng hợp 39 đặc trưng VLE (ngày 0-28)...")
                from src.feature_engineering import generate_features
                generate_features()
                print("\n⏳ [Bước 3/3] Đang chia tập dữ liệu Train (80%) và Test (20%) theo id_student...")
                from src.pipeline_utils import load_and_split_data
                _, X_train, _, X_test, _, _, _ = load_and_split_data(save_split_files=True)
                print("=" * 72)
                print("📂 VỊ TRÍ LƯU CÁC TẬP DỮ LIỆU SAU TIỀN XỬ LÝ:")
                print(f"  • Tập Cohort (27.421 dòng) : {config.COHORT_DATA_PATH}")
                print(f"  • Bảng 39 đặc trưng        : {config.FEATURES_DATA_PATH}")
                print(f"  • Tập dữ liệu TRAIN (80%)  : {config.TRAIN_DATA_PATH} ({len(X_train):,} dòng)")
                print(f"  • Tập dữ liệu TEST  (20%)  : {config.TEST_DATA_PATH} ({len(X_test):,} dòng)")
                print("=" * 72)
                print("✅ Hoàn thành toàn bộ tiền xử lý dữ liệu!")
            elif choice == "2":
                train_submenu()
            elif choice == "3":
                from src.evaluator import run_saved_model_menu
                run_saved_model_menu()
            elif choice == "4":
                from src.compare_models import compare_selected_models_menu
                compare_selected_models_menu()
            elif choice == "5":
                from src.predictor import predict_unlabeled_file_menu
                predict_unlabeled_file_menu()
            elif choice == "6":
                search_student_menu()
            elif choice == "0":
                print("👋 Đã thoát chương trình. Chúc nhóm bảo vệ thành công!")
                sys.exit(0)
            else:
                print("❌ Lựa chọn không hợp lệ, vui lòng nhập từ 0 đến 6!")
        except ImportError as e:
            print(f"⚠️ Chức năng này chưa được nạp đầy đủ ({e}).")
        except Exception as e:
            print(f"\n⚠️ Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    main()