# Customer Segmentation Project

Dự án phân khúc khách hàng sử dụng học máy không giám sát (K-means, có so sánh với GMM và DBSCAN) để phân tích hành vi mua sắm, chia nhóm khách hàng, diễn giải kết quả bằng SHAP, và triển khai thành ứng dụng Streamlit có thể cập nhật dữ liệu và xuất báo cáo.

## Mục tiêu

Phân tích dữ liệu giao dịch của khách hàng để:

- Hiểu rõ hành vi mua sắm của khách hàng qua 16 đặc trưng đa chiều (không chỉ dừng ở RFM truyền thống)
- Chia khách hàng thành các nhóm có đặc điểm tương tự, có cơ sở khoa học trong việc lựa chọn thuật toán
- Diễn giải được vì sao một khách hàng thuộc về một nhóm cụ thể (SHAP + surrogate model)
- Đưa ra chiến lược marketing phù hợp với từng nhóm
- Triển khai thành công cụ thực tế mà doanh nghiệp có thể tự cập nhật dữ liệu và xuất báo cáo

## Dữ liệu

- **Nguồn**: Online Retail Dataset (UCI Machine Learning Repository) — công ty bán lẻ trực tuyến tại UK, chuyên quà tặng và đồ gia dụng, giai đoạn 01/12/2010 – 09/12/2011
- **Dữ liệu gốc**: 541.909 giao dịch, ~4.372 khách hàng (toàn bộ 37 quốc gia, trước khi lọc)
- **Sau làm sạch (UK only — dữ liệu dùng để phân tích chính thức)**: 354.321 giao dịch hợp lệ, **3.920 khách hàng**

## Kết quả chính

- Đã so sánh có hệ thống 3 thuật toán phân cụm (K-Means, GMM, DBSCAN) trên nhiều chỉ số đánh giá (Silhouette, Davies-Bouldin, Calinski-Harabasz) — chọn **K-Means, k = 4** (Silhouette Score = 0,294)
- 4 phân khúc khách hàng: **Giá Trị Thấp (17,4%)**, **Premium (27,0%)**, **Mua Nhiều Thường Xuyên (34,5%)**, **Đa Dạng (21,1%)**
- Kết quả được diễn giải bằng SHAP (qua surrogate Random Forest) và kiểm chứng chéo với Radar Chart
- Triển khai thành dashboard Streamlit gồm 10 trang, bao gồm khả năng tải lên dữ liệu giao dịch mới để tính lại toàn bộ mô hình, và xuất báo cáo dưới 3 định dạng (PDF, Excel, ảnh PNG/JPEG)

## Cấu trúc dự án

```
├── data/
│   ├── raw/                          # Dữ liệu thô
│   └── processed/                    # Dữ liệu đã xử lý (cleaned, features, scaled, clusters)
│       └── *_merged.csv              # Sinh tự động khi dùng tính năng "Cập Nhật Dữ Liệu Mới" trên dashboard
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb     # Làm sạch dữ liệu và EDA
│   ├── 02_feature_engineering.ipynb  # Xây dựng 16 features + Box-Cox + StandardScaler
│   └── 03_modeling.ipynb             # PCA, so sánh thuật toán, K-means, diễn giải SHAP
├── src/
│   └── clustering_library.py         # Thư viện chính (DataCleaner, FeatureEngineer, ClusterAnalyzer, DataVisualizer)
├── docs/
│   └── project_description.md        # Mô tả chi tiết dự án
├── app.py                            # Dashboard Streamlit (10 trang, bao gồm cập nhật dữ liệu & xuất báo cáo)
└── requirements.txt                  # Dependencies
```

## Bắt đầu nhanh

1. **Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

2. **Chạy notebooks theo thứ tự:**
   - `01_cleaning_and_eda.ipynb` — Làm sạch và khám phá dữ liệu
   - `02_feature_engineering.ipynb` — Tạo 16 features + RFM tham chiếu
   - `03_modeling.ipynb` — PCA, so sánh thuật toán phân cụm, K-means, diễn giải SHAP

3. **Chạy Streamlit để xem dashboard:**

```bash
streamlit run app.py
```

   Trong dashboard, có thể tải lên dữ liệu giao dịch mới ở trang **"Cập Nhật Dữ Liệu Mới"** để gộp và tính lại toàn bộ mô hình, hoặc xuất kết quả ở trang **"Xuất Báo Cáo Tổng Hợp"** (PDF / Excel / ảnh PNG-JPEG).

## Công nghệ sử dụng

- **Python**
- **Pandas** — Xử lý dữ liệu
- **Scikit-learn** — Machine learning (K-Means, GMM, DBSCAN, PCA, RandomForest)
- **SciPy** — Box-Cox transformation
- **Matplotlib/Seaborn/Plotly** — Visualization
- **NumPy** — Tính toán số học
- **SHAP** — Giải thích mô hình (explainable AI)
- **Streamlit** — Giao diện dashboard tương tác
- **ReportLab** — Xuất báo cáo PDF (kèm font DejaVu Sans hỗ trợ tiếng Việt)
- **OpenPyXL** — Xuất báo cáo dạng Excel dashboard (KPI động, biểu đồ, bảng có bộ lọc)

## Tài liệu

Chi tiết về phương pháp và lý thuyết được mô tả trong `docs/project_description.md`