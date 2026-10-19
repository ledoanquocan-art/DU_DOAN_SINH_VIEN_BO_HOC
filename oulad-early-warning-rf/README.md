# 🎓 Hệ Thống Cảnh Báo Sớm Nguy Cơ Rút Học Phần (OULAD)

Dự án học thuật ứng dụng Học máy (Machine Learning) để phát hiện sớm sinh viên có nguy cơ bỏ học/rút học phần tại thời điểm **cuối ngày 28** của khóa học. Dự án sử dụng bộ dữ liệu [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

Mô hình dự báo chính của hệ thống là **Random Forest**, được đối chứng với 3 thuật toán: **k-NN, SVM** và **Gaussian Naive Bayes**. Hệ thống được thiết kế dưới dạng giao diện dòng lệnh (CLI) tương tác, cho phép người dùng tùy chỉnh tham số huấn luyện trực tiếp trên Terminal.

---

## 👥 Thành Viên Nhóm 1 (CMC University)

| STT | Họ và Tên | Mã SV | Vai trò & Phụ trách mã nguồn |
|:---:|---|---|---|
| 1 | **Lê Doãn Quốc An** (Leader)| `BCS252485` | Quản lý dự án, Thiết lập lõi (`config.py`, `main.py`), Lọc dữ liệu (`cohort_builder.py`), Tra cứu (`student_lookup.py`). |
| 2 | **Trịnh Phan Tuấn Anh** | `BIT250007` | Trích xuất 39 đặc trưng (`feature_engineering.py`), Pipeline tiện ích, Huấn luyện k-NN (`model_knn.py`). |
| 3 | **Nông Đức Mạnh** | `BAI250041` | Huấn luyện Random Forest & Giải thích đặc trưng (`model_rf.py`), Dự đoán file mới (`predictor.py`). |
| 4 | **Trần Ngọc Tuấn** | `BAI250068` | Huấn luyện SVM (`model_svm.py`), Chạy kiểm thử & Đánh giá (`evaluator.py`). |
| 5 | **Nguyễn Anh Đức** | `BIT253661` | Huấn luyện Naive Bayes (`model_nb.py`), Đối chứng đa mô hình (`compare_models.py`). |

---

## 🚀 Tính Năng Chính Của Hệ Thống

Chương trình điều khiển thông qua file `main.py` với Menu gồm 6 chức năng chính:

1. **Tiền xử lý dữ liệu:** Tự động khảo sát 7 bảng gốc, lọc 27.421 lượt đăng ký hợp lệ tại ngày 28 và tổng hợp 39 đặc trưng hành vi học tập VLE.
2. **Huấn luyện mô hình (Interactive CLI):** Hỏi từng tham số siêu việt (hyperparameters) trước khi huấn luyện (RF, k-NN, SVM, Naive Bayes). Hệ thống chạy 5-Fold CV và lưu tự động file model theo tên tham số.
3. **Chạy kiểm thử (Evaluator):** Chọn mô hình đã lưu để chạy trên tập Test chung (5.465 mẫu). Xuất các chỉ số Accuracy, Macro-F1, Precision, Recall và vẽ Ma trận nhầm lẫn (Confusion Matrix).
4. **Đối chứng đa mô hình (Comparator):** Chọn từ 2 mô hình bất kỳ trở lên để so sánh trực diện. Xuất bảng báo cáo CSV và biểu đồ cột.
5. **Dự đoán dữ liệu mới (Predictor):** Nạp file CSV không nhãn, chọn mô hình để gán nhãn dự đoán nguy cơ và xuất file kết quả.
6. **Tra cứu sinh viên (Lookup):** Nhập mã `id_student` để xem chi tiết 39 thuộc tính nền và hành vi của sinh viên đó.

---

## 📂 Cấu Trúc Thư Mục

```text
DU_DOAN_SINH_VIEN_BO_HOC/
├── data/
│   ├── raw/                  # [Yêu cầu tải] Chứa 7 file CSV gốc của bộ dữ liệu OULAD
│   ├── processed/            # File xuất ra sau Tiền xử lý (cohort_day28.csv, features_day28.csv)
│   └── predictions/          # File CSV dự đoán được sinh ra từ Mục 5
├── results/
│   ├── models/               # Nơi lưu các mô hình .joblib kèm tham số
│   ├── metrics/              # Nơi lưu kết quả đánh giá .json và .csv
│   └── figures/              # Nơi lưu biểu đồ (Ma trận nhầm lẫn, Feature Importance, So sánh)
├── src/
│   ├── __init__.py
│   ├── config.py             # Cấu hình đường dẫn, hằng số và danh sách 39 đặc trưng
│   ├── pipeline_utils.py     # Tiện ích chia tập, chuẩn hóa, nhập tham số & quản lý lưu file
│   ├── cohort_builder.py     # Lọc dữ liệu nhóm dự báo
│   ├── feature_engineering.py# Trích xuất đặc trưng VLE
│   ├── model_rf.py           # Huấn luyện Random Forest
│   ├── model_knn.py          # Huấn luyện k-NN
│   ├── model_svm.py          # Huấn luyện SVM
│   ├── model_nb.py           # Huấn luyện Naive Bayes
│   ├── evaluator.py          # Đánh giá & xuất biểu đồ
│   ├── compare_models.py     # Đối chứng mô hình
│   ├── predictor.py          # Dự đoán dữ liệu chưa có nhãn
│   └── student_lookup.py     # Tra cứu hồ sơ sinh viên
├── main.py                   # Điểm khởi chạy chương trình (Menu chính)
├── requirements.txt          # Các thư viện phụ thuộc
└── .gitignore                # Chặn upload file dữ liệu nặng/model

⚙️ Hướng Dẫn Cài Đặt & Sử Dụng
1. Chuẩn bị môi trường
Yêu cầu Python 3.9 trở lên. Nên sử dụng môi trường ảo (virtual environment).
bash
cd DU_DOAN_SINH_VIEN_BO_HOC

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

2. Chuẩn bị dữ liệu gốc
Truy cập Kaggle hoặc trang chủ OULAD để tải bộ dữ liệu anonymisedData.zip.

Giải nén toàn bộ 7 file .csv (studentInfo.csv, studentVle.csv, vle.csv,...) vào thư mục data/raw/ của dự án. (Thư mục này đã được định cấu hình bỏ qua trong .gitignore nên an toàn không bị đẩy file nặng lên GitHub).

3. Chạy chương trình
Khởi động hệ thống từ Terminal:

Bash
python main.py
Lần chạy đầu tiên: Bạn phải chọn 1 (Chạy tiền xử lý dữ liệu) để hệ thống đọc file dữ liệu lớn và sinh ra bảng 39 đặc trưng. Mất khoảng 1-2 phút cho bước này.

Tiếp theo: Chọn 2 để huấn luyện các mô hình. Hãy mạnh dạn thay đổi tham số theo lời nhắc trên màn hình. Các cấu hình sẽ được lưu độc lập.

Cuối cùng: Chọn 3, 4, hoặc 5 để đánh giá, đối chứng và dự đoán.

🛠 Công Nghệ Sử Dụng
Ngôn ngữ: Python

Xử lý dữ liệu: pandas, numpy

Học máy (Machine Learning): scikit-learn

Trực quan hóa: matplotlib

Quản lý quy trình làm việc: Git, GitHub
