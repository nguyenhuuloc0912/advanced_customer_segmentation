# Mô tả Chi tiết Dự án Customer Segmentation

## Giới thiệu tổng quan về dự án

Dự án **Customer Segmentation** (Phân khúc khách hàng) là một ứng dụng thực tế của machine learning trong lĩnh vực business intelligence và marketing analytics. Dự án này tập trung vào việc phân tích hành vi mua sắm của khách hàng từ dữ liệu giao dịch thực tế để chia họ thành các nhóm có đặc điểm tương tự nhau.

Mục tiêu chính của dự án là xây dựng một hệ thống tự động có thể:

- Phân tích patterns trong dữ liệu giao dịch của khách hàng
- Chia khách hàng thành các segments có ý nghĩa business
- Cung cấp insights để tối ưu hóa chiến lược marketing và customer relationship management
- Diễn giải được (interpretability) lý do một khách hàng thuộc về một segment cụ thể, thay vì chỉ dừng lại ở việc gán nhãn
- Triển khai kết quả thành một ứng dụng thực tế (Streamlit Dashboard) mà doanh nghiệp có thể vận hành, cập nhật dữ liệu và xuất báo cáo

## Đặt vấn đề và bài toán business

Trong thời đại số hóa hiện nay, các doanh nghiệp bán lẻ đối mặt với thách thức lớn là **làm sao hiểu rõ khách hàng của mình** để có thể:

### Vấn đề thực tế:

1. **Chi phí marketing ngày càng tăng** - Cần tối ưu hóa budget cho đúng đối tượng
2. **Customer acquisition cost cao** - Cần tập trung vào retention thay vì chỉ acquisition
3. **Cạnh tranh gay gắt** - Cần differentiation thông qua personalization
4. **Dữ liệu khách hàng phong phú nhưng chưa khai thác hiệu quả**

### Bài toán business cụ thể:

- **"Làm sao để chia 3.920 khách hàng (thị trường UK) thành các nhóm có ý nghĩa?"**
- **"Nhóm khách hàng nào đáng đầu tư nhất?"**
- **"Chiến lược marketing nào phù hợp với từng nhóm?"**

### Giải pháp đề xuất:

Sử dụng **unsupervised machine learning** để tự động phát hiện patterns ẩn trong dữ liệu, chia khách hàng thành các segments dựa trên hành vi mua sắm thực tế, và diễn giải kết quả bằng các kỹ thuật explainable AI (SHAP) để đảm bảo kết quả không chỉ chính xác mà còn có thể giải thích được cho người ra quyết định kinh doanh.

## Giới thiệu về Supervised và Unsupervised Learning

### Supervised Learning (Học có giám sát)

**Định nghĩa**: Là phương pháp machine learning sử dụng dữ liệu có nhãn (labeled data) để training model.

**Đặc điểm**:

- Có target variable rõ ràng (y)
- Model học từ input-output pairs
- Mục tiêu: dự đoán output cho input mới

**Ví dụ**:

- Classification: Dự đoán email có phải spam không
- Regression: Dự đoán giá nhà dựa trên diện tích, vị trí

**Công thức tổng quát**: `f(X) = y`

### Unsupervised Learning (Học không giám sát)

**Định nghĩa**: Là phương pháp machine learning tìm patterns ẩn trong dữ liệu **không có nhãn**.

**Đặc điểm**:

- Không có target variable
- Model tự khám phá cấu trúc dữ liệu
- Mục tiêu: tìm patterns, groups, associations

**Các loại chính**:

1. **Clustering**: Chia dữ liệu thành nhóm (K-means, GMM, DBSCAN, Hierarchical)
2. **Association Rules**: Tìm mối quan hệ (Market Basket Analysis)
3. **Dimensionality Reduction**: Giảm chiều dữ liệu (PCA, t-SNE)

### Tại sao chọn Unsupervised Learning cho bài toán này?

**Lý do chính**:

- **Không có ground truth**: Chúng ta không biết trước khách hàng thuộc nhóm nào
- **Khám phá tự nhiên**: Muốn để dữ liệu "nói" về các patterns tự nhiên
- **Flexibility**: Không bị constraint bởi định nghĩa nhóm từ trước
- **Scalability**: Có thể áp dụng cho bất kỳ dataset nào

## Cách tiếp cận

### 1. Comprehensive Customer Behavior Analysis

Thay vì chỉ sử dụng framework RFM truyền thống, dự án này áp dụng phương pháp **phân tích đa chiều** với **16 features** ở cấp độ khách hàng để nắm bắt toàn diện hành vi mua sắm. RFM chỉ được sử dụng như **phương án tham khảo** để trực quan hóa và đối chiếu kết quả, không được dùng trực tiếp làm đầu vào cho mô hình phân cụm.

### 2. Pipeline xử lý dữ liệu

```
Raw Data → Data Cleaning → Feature Engineering → Transformation (Box-Cox + StandardScaler)
        → Giảm chiều (PCA) → Clustering (so sánh K-Means / GMM / DBSCAN) → Diễn giải (SHAP) → Validation
```

**Chi tiết từng bước**:

1. **Data Cleaning**:

   - Loại bỏ giao dịch hủy (InvoiceNo bắt đầu bằng 'C')
   - Focus vào khách hàng UK
   - Xử lý missing values (loại bản ghi thiếu CustomerID)

2. **Feature Engineering**:

   - Tạo 16 customer-level features toàn diện
   - Aggregate transaction data với multiple perspectives (theo hóa đơn, theo sản phẩm)
   - RFM analysis như reference cho visualization và đối chiếu

3. **Data Transformation**:

   - Box-Cox transformation cho distribution normalization
   - StandardScaler cho feature scaling

4. **Dimensionality Reduction**:

   - PCA (Principal Component Analysis), giữ lại số thành phần chính đảm bảo tối thiểu 85% tổng phương sai giải thích
   - Mục đích: loại bỏ đa cộng tuyến giữa 16 đặc trưng gốc và giảm nhiễu trước khi phân cụm

5. **Clustering**:

   - So sánh có hệ thống ba thuật toán: K-means, Gaussian Mixture Models (GMM), DBSCAN
   - Đánh giá bằng ba chỉ số: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
   - Xác định k tối ưu bằng Elbow method và Silhouette analysis

6. **Diễn giải (Explainability)**:

   - Huấn luyện mô hình surrogate Random Forest để dự đoán lại nhãn cụm từ 16 đặc trưng gốc
   - Áp dụng SHAP (TreeExplainer) lên mô hình surrogate để xác định đặc trưng nào chi phối việc phân định từng cụm
   - Đối chiếu chéo với Radar Chart để kiểm chứng tính nhất quán của kết quả diễn giải

7. **Validation**:

   - Đánh giá định lượng đa chỉ số (không chỉ dựa vào một chỉ số duy nhất)
   - Business interpretation của các clusters, đối chiếu với khung RFM tham chiếu

## Chi tiết về dữ liệu

### Nguồn dữ liệu

- **Dataset**: Online Retail Data from UCI Machine Learning Repository
- **Công ty**: UK-based non-store online retail
- **Ngành**: Quà tặng và đồ gia dụng
- **Thời gian**: 01/12/2010 - 09/12/2011
- **Địa lý**: Chủ yếu UK, một phần châu Âu và toàn cầu (37 quốc gia trong dữ liệu gốc)

### Cấu trúc dữ liệu raw

| Column      | Type     | Description                           | Example             |
| ----------- | -------- | ------------------------------------- | -------------------- |
| InvoiceNo   | object   | Mã hóa đơn (6 chữ số, 'C' = canceled) | 536365              |
| StockCode   | object   | Mã sản phẩm (5 chữ số)                | 85123A              |
| Description | object   | Tên sản phẩm                          | WHITE HANGING HEART |
| Quantity    | int64    | Số lượng sản phẩm                     | 6                   |
| InvoiceDate | datetime | Thời gian giao dịch                   | 2010-12-01 08:26:00 |
| UnitPrice   | float64  | Đơn giá (GBP)                         | 2.55                |
| CustomerID  | object   | ID khách hàng (5-6 chữ số)            | 17850               |
| Country     | object   | Quốc gia khách hàng                   | United Kingdom      |

### Thống kê mô tả

**Dữ liệu gốc**:

- **Tổng giao dịch**: 541.909 records
- **Khách hàng unique**: ~4.372 customers (toàn bộ 37 quốc gia, trước khi lọc)
- **Sản phẩm unique**: ~4.000 products
- **Quốc gia**: 37 countries

**Sau làm sạch (UK only) — dữ liệu chính thức dùng cho mô hình**:

- **Giao dịch hợp lệ**: 354.321 records (loại 187.588 giao dịch, tương đương 34,6%)
- **Khách hàng**: 3.920 customers
- **Thời gian**: 374 ngày (01/12/2010 – 09/12/2011)
- **Chi tiêu trung bình/khách**: £1.864,39 (trung vị £652,28)
- **Số giao dịch trung bình/khách**: 4,2 lần

### Đặc điểm dữ liệu

**Challenges**:

1. **Missing CustomerID**: khoảng 25% giao dịch trong dữ liệu gốc không có CustomerID
2. **Negative Quantity**: Giao dịch hủy/hoàn trả
3. **Extreme values**: Một số giao dịch có giá trị rất cao
4. **Skewed distribution**: Phân phối lệch phải của hầu hết các đặc trưng hành vi khách hàng — lý do chính khiến Box-Cox transformation là bước bắt buộc trước khi phân cụm

**Opportunities**:

1. **Rich transactional data**: Có thông tin chi tiết về việc chi tiêu của khách hàng
2. **Time series**: Có thể phân tích trends theo thời gian
3. **Product diversity**: Nhiều categories sản phẩm

## Phương hướng xây dựng Feature Engineering

### 1. Comprehensive Customer Feature Set

Thay vì chỉ sử dụng RFM truyền thống, dự án này xây dựng **16 features toàn diện** ở cấp độ khách hàng để nắm bắt đầy đủ các khía cạnh hành vi mua sắm. RFM chỉ được sử dụng như **phương án tham khảo** để trực quan hóa và đối chiếu, được tính toán song song, độc lập với tập đặc trưng chính.

### Mô tả Features

16 features ở cấp độ khách hàng, chia thành bốn nhóm bổ sung cho nhau:

**Nhóm cơ bản (6 features):**

1. `Sum_Quantity`: Tổng số lượng sản phẩm đã mua
2. `Mean_UnitPrice`: Giá trung bình trên mỗi đơn vị trong tất cả lần mua
3. `Mean_TotalPrice`: Số tiền trung bình mỗi giao dịch
4. `Sum_TotalPrice`: Tổng số tiền đã chi (giá trị vòng đời khách hàng)
5. `Count_Invoice`: Số lượng giao dịch (hóa đơn) duy nhất
6. `Count_Stock`: Số lượng sản phẩm duy nhất đã mua

**Nhóm tổng hợp theo sản phẩm (2 features):**

7. `Mean_InvoiceCountPerStock`: Tần suất mua trung bình trên mỗi sản phẩm
8. `Mean_StockCountPerInvoice`: Số lượng sản phẩm khác nhau trung bình mỗi giao dịch

**Nhóm tổng hợp theo hóa đơn (4 features):**

9. `Mean_UnitPriceMeanPerInvoice`: Giá đơn vị trung bình mỗi giao dịch
10. `Mean_QuantitySumPerInvoice`: Số lượng trung bình mỗi giao dịch
11. `Mean_TotalPriceMeanPerInvoice`: Số tiền trung bình mỗi sản phẩm trong giao dịch
12. `Mean_TotalPriceSumPerInvoice`: Tổng chi tiêu trung bình mỗi giao dịch

**Nhóm tổng hợp theo loại sản phẩm (4 features):**

13. `Mean_UnitPriceMeanPerStock`: Mức giá trung bình trên mỗi sản phẩm
14. `Mean_QuantitySumPerStock`: Số lượng trung bình đã mua trên mỗi sản phẩm
15. `Mean_TotalPriceMeanPerStock`: Chi tiêu trung bình trên mỗi sản phẩm
16. `Mean_TotalPriceSumPerStock`: Tổng chi tiêu trung bình trên mỗi sản phẩm

### 2. Ý nghĩa Business của Features

#### Nhóm Chỉ số cơ bản (1-6)

Các features này cung cấp **overview tổng quan** về quy mô và giá trị của khách hàng:

- **Volume indicators**: Sum_Quantity, Count_Invoice, Count_Stock
- **Value indicators**: Sum_TotalPrice, Mean_UnitPrice, Mean_TotalPrice
- **Diversity indicators**: Count_Stock cho thấy tính đa dạng sản phẩm

#### Nhóm Tổng hợp theo sản phẩm (7-8)

Features này phản ánh **loyalty và engagement patterns**:

- `Mean_InvoiceCountPerStock`: Khách hàng có xu hướng mua lại sản phẩm không?
- `Mean_StockCountPerInvoice`: Khách hàng mua tập trung hay đa dạng mỗi lần?

#### Nhóm Tổng hợp theo hóa đơn (9-12)

Features này mô tả **transaction behavior**:

- Consistency trong spending per transaction
- Average basket composition
- Price sensitivity patterns

#### Nhóm Tổng hợp theo loại sản phẩm (13-16)

Features này cho thấy **product preferences**:

- Preference cho premium vs budget products
- Quantity buying patterns per product type
- Category-specific spending behavior

## Giới thiệu về Box-Cox Transformation

### Tại sao cần transformation?

**Vấn đề với raw customer behavior data**:

1. **Skewed distribution**: 16 features thường có phân phối lệch phải do `nature` của business data
2. **Different scales**: Quantity (units), Price (currency), Count (numbers) có scale khác nhau
3. **Outliers**: High-value customers tạo ra extreme values
4. **Clustering sensitivity**: K-means nhạy cảm với scale và distribution differences

### Box-Cox Transformation là gì?

**Định nghĩa**: Box-Cox là một family of power transformations để normalize distribution.

**Công thức**:

```
y(λ) = (x^λ - 1) / λ     if λ ≠ 0
y(λ) = ln(x)             if λ = 0
```

**Tham số λ**:

- **λ = 1**: Không transformation (identity)
- **λ = 0.5**: Square root transformation
- **λ = 0**: Log transformation
- **λ = -0.5**: Inverse square root
- **λ = -1**: Inverse transformation

**Process**:

1. **Handle zeros/negatives**: Shift data if needed
2. **MLE optimization**: Tìm λ maximize likelihood
3. **Apply transformation**: Transform với λ tối ưu
4. **Validate normality**: Kiểm tra distribution sau transform

### Lợi ích của Box-Cox

**Statistical benefits**:

- **Normalization**: Đưa skewed data về gần normal
- **Variance stabilization**: Giảm heteroscedasticity
- **Linearity improvement**: Tăng linear relationships

**Machine learning benefits**:

- **Better clustering**: K-means hoạt động tốt hơn với normal data
- **Reduced outlier impact**: Transform làm giảm extreme values
- **Improved convergence**: Algorithms converge nhanh hơn

### Thực hiện trong dự án

**Steps**:

1. **Shift features**: Ensure all values > 0
2. **Find optimal λ**: Cho từng feature riêng biệt
3. **Apply transformation**: Transform each feature
4. **Standardization**: StandardScaler sau transformation
5. **Giảm chiều**: PCA áp dụng sau cùng, giữ lại số thành phần chính đạt tối thiểu 85% phương sai giải thích, trước khi đưa vào các thuật toán phân cụm

## Chi tiết về Clustering: so sánh K-means, GMM và DBSCAN

### K-means Algorithm Overview

**Định nghĩa**: K-means là thuật toán clustering phổ biến nhất, chia n observations thành k clusters sao cho mỗi observation thuộc cluster có mean gần nhất.

### Nguyên lý hoạt động

#### 1. Objective Function

K-means minimize **Within-Cluster Sum of Squares (WCSS)**:

```
WCSS = Σ(i=1 to k) Σ(x∈Ci) ||x - μi||²
```

Trong đó:

- **k**: số clusters
- **Ci**: cluster thứ i
- **μi**: centroid của cluster i
- **x**: data point

#### 2. Algorithm Steps

**Initialization**:

```python
# Random initialization của k centroids
centroids = randomly_select_k_points(data)
```

**Iterative Process**:

```
Repeat until convergence:
    1. Assignment Step:
       - Assign mỗi point tới centroid gần nhất
       - cluster[i] = argmin(distance(point[i], centroid[j]))

    2. Update Step:
       - Update centroid = trung bình của assigned points
       - centroid[j] = mean(points in cluster[j])
```

**Convergence criteria**:

- Centroids không thay đổi
- Assignments không thay đổi
- WCSS improvement < threshold

#### 3. Distance Metrics

**Euclidean Distance (default)**:

```
d(x,y) = √(Σ(xi - yi)²)
```

### Ưu điểm và hạn chế

#### Ưu điểm

1. **Simple & Fast**: O(nkt) complexity
2. **Scalable**: Hoạt động tốt với large datasets
3. **Interpretable**: Centroids có ý nghĩa business rõ ràng
4. **Deterministic**: Với same initialization (random_state cố định), same result

#### Hạn chế

1. **Require pre-defined k**: Cần biết số clusters trước
2. **Sensitive to initialization**: Different starts → different results
3. **Assume spherical clusters**: Không tốt với irregular shapes
4. **Sensitive to outliers**: Outliers kéo lệch centroids

### Cách xác định k tối ưu

#### 1. Elbow Method

```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    wcss.append(kmeans.inertia_)

# Plot và tìm "elbow point"
```

**Nguyên lý**: Tìm điểm mà WCSS giảm chậm lại (diminishing returns)

#### 2. Silhouette Analysis

```python
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)
```

**Silhouette coefficient**:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Trong đó:

- **a(i)**: Khoảng cách trung bình đến points trong cùng cluster
- **b(i)**: Khoảng cách trung bình đến points trong cluster gần nhất

**Interpretation**:

- **s close to 1**: Point thuộc đúng cluster
- **s close to 0**: Point ở boundary
- **s negative**: Point có thể thuộc cluster khác

### So sánh với các thuật toán khác: GMM và DBSCAN

Thay vì chỉ dùng K-means, dự án so sánh có hệ thống ba thuật toán trên cùng một không gian đặc trưng đã giảm chiều bằng PCA, với ba chỉ số Silhouette Score, Davies-Bouldin Index và Calinski-Harabasz Index:

| Thuật toán       | Cấu hình                    | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
| ---------------- | ---------------------------- | ------------- | ----------------- | -------------------- |
| **K-Means**       | **k = 4 (được chọn)**        | **0,294**     | **1,139**          | 1.431,1              |
| K-Means           | k = 3                         | 0,286         | 1,225              | **1.462,9**           |
| GMM (diag)         | k = 4                         | 0,237         | 1,904              | 857,0                 |
| GMM (full)         | k = 4                         | 0,197         | 1,828              | 897,3                 |
| DBSCAN             | eps=0,8; min_samples=5 (5 cụm) | 0,160         | 0,595              | 21,7                  |

**Kết luận lựa chọn**: K-means với k=4 vượt trội trên hai trong ba chỉ số (Silhouette, Davies-Bouldin), trong khi GMM và DBSCAN đều cho kết quả kém hơn rõ rệt — phù hợp với đặc điểm cấu trúc dữ liệu hành vi khách hàng sau PCA có xu hướng gần với giả định cụm hình cầu của K-means hơn là giả định mật độ (DBSCAN) hay hỗn hợp Gaussian chồng lấn (GMM). Giữa k=3 và k=4, mặc dù Calinski-Harabasz Index hơi nghiêng về k=3, việc phân tích sâu cấu trúc phân cụm cho thấy k=4 tách được nhóm Premium rộng của k=3 thành hai nhóm có ý nghĩa hành vi khác biệt rõ rệt (Premium thật và Giá trị thấp) — đây là căn cứ quyết định chọn k=4 làm phương án chính thức.

### Implementation trong dự án

#### Hyperparameter tuning:

```python
kmeans = KMeans(
    n_clusters=4,          # Optimal k từ Elbow + Silhouette + phân tích cấu trúc phân cụm
    init='k-means++',      # Smart initialization
    n_init=10,            # Multiple runs
    max_iter=300,         # Convergence limit
    random_state=42       # Reproducibility
)
```

#### Validation process:

1. **Internal validation**: Silhouette, Davies-Bouldin, Calinski-Harabasz (đa chỉ số, không chỉ một)
2. **External validation**: Business interpretation, đối chiếu với RFM tham chiếu
3. **Explainability validation**: SHAP (qua surrogate Random Forest) đối chiếu chéo với Radar Chart
4. **Stability validation**: Multiple runs với random_state cố định

### Diễn giải mô hình bằng SHAP (Explainable AI)

Vì K-means (như hầu hết thuật toán phân cụm) không có cơ chế diễn giải trực tiếp mức độ đóng góp của từng đặc trưng, dự án áp dụng chiến lược **surrogate model**:

1. Huấn luyện một mô hình Random Forest (có giám sát) để dự đoán lại nhãn cụm đã có từ K-means, dựa trên 16 đặc trưng gốc
2. Đánh giá độ chính xác của mô hình đại diện này trên tập kiểm tra độc lập, đảm bảo đủ tin cậy trước khi dùng cho mục đích diễn giải
3. Áp dụng SHAP TreeExplainer lên mô hình Random Forest để tính đóng góp của từng đặc trưng vào việc phân định từng cụm cụ thể
4. Trực quan hóa bằng SHAP Beeswarm Chart, tách riêng theo từng cụm (0/1/2/3 cho k=4)
5. Đối chiếu chéo kết quả SHAP với Radar Chart (biểu diễn giá trị trung bình đặc trưng từng cụm) để kiểm chứng tính nhất quán

### Business Interpretation của Clusters (kết quả thực tế, K-means k=4)

**Bốn phân khúc khách hàng đã được xác định** (dựa trên toàn bộ 3.920 khách hàng UK):

1. **Nhóm Giá Trị Thấp** — 17,4% (682 khách hàng)

   - Chi tiêu, tần suất giao dịch và mức độ đa dạng sản phẩm đều ở mức thấp nhất
   - Đặc trưng SHAP chi phối: không nổi bật riêng lẻ — kết hợp của đơn giá và tổng giá trị giao dịch ở mức thấp
   - Strategy: Ưu đãi kích hoạt lại (win-back), khảo sát nguyên nhân giảm tương tác

2. **Nhóm Premium** — 27,0% (1.059 khách hàng)

   - Đơn giá trung bình cao nhất trong toàn bộ các cụm (Mean_UnitPrice ≈ 8,90)
   - Tần suất giao dịch không cao, nhưng ưu tiên sản phẩm giá trị cao
   - Strategy: Chương trình VIP, quyền tiếp cận sớm sản phẩm mới, hạn chế giảm giá trực tiếp

3. **Nhóm Mua Nhiều Thường Xuyên** — 34,5% (1.352 khách hàng, nhóm lớn nhất)

   - Tổng số lượng và tổng giá trị mua sắm cao nhất (Sum_Quantity, Sum_TotalPrice)
   - Tần suất giao dịch cao, đóng góp phần lớn doanh thu tổng thể
   - Strategy: Chương trình tích điểm theo cấp bậc, mô hình subscription định kỳ

4. **Nhóm Đa Dạng** — 21,1% (827 khách hàng)

   - Mức độ đa dạng sản phẩm cao (Count_Stock, Mean_StockCountPerInvoice)
   - Xu hướng khám phá nhiều chủng loại sản phẩm thay vì tập trung vào một số ít
   - Strategy: Hệ thống gợi ý sản phẩm (recommendation engine), cross-selling cá nhân hóa

**RFM Reference Validation**:

- Đối chiếu bốn phân khúc trên với khung RFM tham chiếu cho thấy sự tương đồng với nhóm Mua Nhiều Thường Xuyên (Frequency/Monetary cao) và nhóm Giá Trị Thấp (Recency cao, Frequency/Monetary thấp)
- Điểm khác biệt quan trọng: nhóm Premium và nhóm Đa Dạng là hai phân khúc mà RFM truyền thống (chỉ 3 chiều) không thể phân tách rõ ràng như tập 16 đặc trưng — đây là minh chứng cho giá trị gia tăng của việc mở rộng không gian đặc trưng

## Ứng dụng triển khai: Streamlit Dashboard

Toàn bộ pipeline nghiên cứu được triển khai thành một ứng dụng dashboard tương tác trên nền tảng Streamlit, gồm mười trang chức năng:

1. **Tổng Quan** — thống kê tổng thể về dữ liệu và phân khúc
2. **Khám Phá Phân Khúc** — Radar Chart chi tiết đặc trưng từng cụm
3. **So Sánh K=3 vs K=4** — biểu đồ Sankey minh họa cách cụm Premium ở k=3 tách thành hai cụm ở k=4
4. **Phân Tích RFM** — khung tham chiếu RFM đối chiếu với kết quả phân cụm chính
5. **Không Gian PCA** — trực quan hóa vị trí khách hàng trong không gian đã giảm chiều
6. **Xếp Hạng Khách Hàng** — bảng xếp hạng theo các tiêu chí giá trị khác nhau
7. **Tra Cứu Khách Hàng** — tra cứu thông tin/phân khúc theo mã khách hàng cụ thể
8. **Mô Phỏng Khách Hàng (What-if Simulator)** — dự đoán phân khúc theo thời gian thực cho khách hàng giả định
9. **Cập Nhật Dữ Liệu Mới** — cho phép doanh nghiệp tải lên giao dịch mới, tự động làm sạch, gộp với dữ liệu hiện có, và **huấn luyện lại toàn bộ pipeline** (Box-Cox, PCA, K-means) trên tập dữ liệu đã gộp
10. **Xuất Báo Cáo Tổng Hợp** — xuất kết quả phân tích dưới 3 định dạng: PDF (trình bày/báo cáo), Excel (dashboard có bộ lọc và biểu đồ để phân tích thêm), và ảnh PNG/JPEG (chèn nhanh vào tài liệu/slide)

Việc tái sử dụng trực tiếp các thành phần xử lý đã được kiểm chứng trong nghiên cứu vào ứng dụng triển khai đảm bảo tính nhất quán giữa kết quả học thuật và công cụ thực tiễn.

## Giới thiệu về Notebooks và Source Code

### Cấu trúc Source Code

#### 1. clustering_library.py - Core Library

**Class DataCleaner**:

```python
class DataCleaner:
    """Xử lý data cleaning và basic EDA"""

    def load_data()          # Load và format data
    def clean_data()         # Remove invalid records
    def explore_customers()  # Customer-level analysis
```

**Class FeatureEngineer**:

```python
class FeatureEngineer:
    """Feature engineering và transformations"""

    def create_customer_features()  # Generate 16 comprehensive features
    def create_rfm_reference()      # RFM cho visualization/tham chiếu
    def transform_features()        # Box-Cox + StandardScaler
```

**Class ClusterAnalyzer**:

```python
class ClusterAnalyzer:
    """Clustering, so sánh thuật toán, và diễn giải"""

    def find_optimal_clusters()   # Elbow + silhouette
    def compare_clustering()      # So sánh K-Means / GMM / DBSCAN
    def fit_kmeans()               # Train mô hình chính thức
    def train_surrogate_model()   # Random Forest surrogate cho SHAP
    def plot_shap_summary()        # SHAP Beeswarm theo từng cụm
```

**Class DataVisualizer**:

```python
class DataVisualizer:
    """Visualization và reporting"""

    def plot_missing_data()               # Missing value heatmap
    def plot_sales_trends()               # Time series analysis
    def plot_customer_distribution()      # Customer distribution
    def plot_features_boxplots()          # 16 features trước/sau Box-Cox
    def plot_rfm_analysis()               # Phân phối RFM
```

#### 2. Design Patterns

**Object-Oriented Design**:

- Mỗi class có responsibility rõ ràng
- Encapsulation của methods và attributes
- Reusability cho different datasets

**Pipeline Pattern**:

```python
# Chuỗi xử lý có thể compose
cleaner = DataCleaner(data_path)
engineer = FeatureEngineer()
analyzer = ClusterAnalyzer()

# Pipeline execution
df_clean = cleaner.clean_data()
features = engineer.fit_transform(df_clean)
segments = analyzer.fit_predict(features)
```

### Chi tiết về từng Notebook

#### 1. 01_cleaning_and_eda.ipynb

**Mục tiêu**: Làm sạch dữ liệu và khám phá ban đầu

**Sections**:

1. **Data Loading & Overview**

   - Load 541.909 giao dịch
   - Data types và memory usage
   - Missing value analysis

2. **Data Cleaning Process**

   - Remove canceled orders (C prefix)
   - Focus on UK customers only
   - Handle missing CustomerIDs
   - Result: **354.321 giao dịch hợp lệ** (3.920 khách hàng UK)

3. **Exploratory Data Analysis**

   - Sales trends theo thời gian
   - Customer transaction patterns
   - Product analysis
   - Phân phối RFM tham chiếu

4. **Key Insights Discovery**
   - Seasonal patterns (tăng mạnh vào tháng 11 — mùa mua sắm cuối năm)
   - Customer behavior patterns
   - Data quality assessment

**Outputs**: Clean dataset ready for feature engineering

#### 2. 02_feature_engineering.ipynb

**Mục tiêu**: Tạo customer-level features cho clustering

**Sections**:

1. **16 Features Creation**

   - Aggregate transaction data theo 4 nhóm (cơ bản, sản phẩm, hóa đơn, loại sản phẩm)
   - RFM tính song song làm tham chiếu

2. **Distribution Analysis**

   - Feature distributions visualization
   - Skewness và outlier detection

3. **Data Transformation**
   - Box-Cox transformation cho normality
   - StandardScaler cho equal weights
   - Validation của transformation quality (boxplot/histogram trước và sau)

**Outputs**: Ma trận đặc trưng đã chuẩn hóa, sẵn sàng cho PCA và clustering

#### 3. 03_modeling.ipynb

**Mục tiêu**: Giảm chiều, xây dựng, so sánh và diễn giải mô hình phân cụm

**Sections**:

1. **PCA** — giảm chiều, giữ ≥85% phương sai giải thích

2. **Optimal Clusters Selection**

   - Elbow method + Silhouette analysis (k = 2 → 10)

3. **So sánh thuật toán**

   - K-means (k=3, k=4), GMM (diag, full), DBSCAN
   - Đánh giá bằng Silhouette, Davies-Bouldin, Calinski-Harabasz

4. **Cluster Analysis**

   - Radar Chart đặc trưng từng cụm
   - Business meaning của từng cluster

5. **Diễn giải bằng SHAP**

   - Surrogate Random Forest + TreeExplainer
   - SHAP Beeswarm theo từng cụm

6. **Results Visualization**
   - Biểu đồ phân cụm PCA 2D/3D
   - Bảng so sánh thuật toán

**Outputs**: Mô hình phân cụm chính thức (K-means k=4) và toàn bộ kết quả diễn giải

### Code Quality & Best Practices

#### 1. Documentation

```python
def create_customer_features(self, df):
    """
    Tạo 16 customer-level features từ transaction data.

    Args:
        df (pd.DataFrame): Transaction data đã làm sạch với columns
                          [CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice]

    Returns:
        pd.DataFrame: Customer features (16 cột), index = CustomerID

    Example:
        >>> features = engineer.create_customer_features(df_clean)
        >>> print(features.shape)
        (3920, 16)
    """
```

#### 2. Error Handling

```python
def load_data(self):
    try:
        self.df = pd.read_csv(self.data_path, encoding="latin1")
        assert len(self.df) > 0, "Dataset is empty"
        return self.df
    except FileNotFoundError:
        raise ValueError(f"Data file not found: {self.data_path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")
```

#### 3. Configurability

```python
# Parameters có thể tune
CONFIG = {
    'clustering': {
        'max_clusters': 10,
        'init_method': 'k-means++',
        'n_init': 10,
        'random_state': 42
    },
    'transformation': {
        'method': 'boxcox',
        'scaler': 'standard'
    },
    'pca': {
        'variance_threshold': 0.85
    }
}
```

## Tổng kết

### Thành quả đạt được

#### 1. Technical Achievements

- **Automated Pipeline**: Xây dựng được pipeline tự động, có thể tái lập, từ raw data đến final segments
- **Robust Preprocessing**: Data cleaning và feature engineering chất lượng cao, đã kiểm chứng và sửa các lỗi kỹ thuật quan trọng (ví dụ: lỗi rò rỉ nhãn cụm vào ma trận PCA từng gây Silhouette Score ảo)
- **So sánh đa thuật toán có cơ sở khoa học**: K-means, GMM, DBSCAN — không chọn mô hình theo cảm tính
- **Explainable AI cho bài toán phân cụm**: tích hợp SHAP thông qua surrogate model, kiểm chứng chéo với Radar Chart

#### 2. Business Impact

- **Customer Insights**: Hiểu rõ 4 nhóm khách hàng chính với đặc điểm hành vi riêng biệt, có thể diễn giải
- **Actionable Segments**: Mỗi segment có chiến lược marketing cụ thể, khả thi để triển khai
- **Data-Driven Decisions**: Foundation cho personalized marketing campaigns
- **Ứng dụng thực tế**: Dashboard vận hành được, có khả năng cập nhật dữ liệu mới và xuất báo cáo đa định dạng, không chỉ dừng ở notebook nghiên cứu

#### 3. Model Performance (kết quả cuối cùng, đã kiểm chứng)

```
Kết quả phân cụm chính thức — K-means, k = 4:
- Silhouette Score:        0,294
- Davies-Bouldin Index:    1,139
- Calinski-Harabasz Index: 1.431,1
- 4 clusters: 17,4% / 27,0% / 34,5% / 21,1%
- Đã kiểm chứng chéo bằng Radar Chart + SHAP
- Reproducible với random_state cố định
```

### Hướng phát triển tiếp theo

#### 1. Model Enhancement

- **Thử nghiệm thêm thuật toán chưa so sánh**: Hierarchical Clustering, HDBSCAN (biến thể nâng cao của DBSCAN, xử lý tốt hơn dữ liệu mật độ không đồng đều)
- **Mở rộng dữ liệu**: nhiều thị trường quốc tế hơn, khoảng thời gian quan sát dài hơn
- **Dynamic segmentation**: phân cụm động (dynamic clustering), theo dõi sự dịch chuyển khách hàng giữa các phân khúc theo thời gian
- **Ensemble methods**: kết hợp nhiều phương pháp phân cụm

#### 2. Business Applications

- **Recommendation System**: Product recommendations cho từng segment
- **Price Optimization**: Dynamic pricing based on segments
- **Churn Prediction**: Supervised learning cho at-risk customers
- **CLV Modeling**: Customer Lifetime Value prediction
- **Dashboard thời gian thực**: mở rộng khả năng theo dõi biến động phân khúc khi có dữ liệu giao dịch mới liên tục
- **Ứng dụng Multi-Agents**: Sử dụng đặc trưng của từng segment để mô phỏng lại hành vi khách hàng bằng AI agents, dùng thảo luận giữa multi-agents để đưa ra chiến lược marketing cá nhân hóa hơn cho từng segment

### Kết luận cuối cùng

Dự án **Customer Segmentation** này đã:

1. **Chuyển đổi raw transaction data (541.909 giao dịch) thành actionable business insights** (4 phân khúc trên 3.920 khách hàng UK, đã kiểm chứng và diễn giải được)
2. **Xây dựng automated pipeline có thể áp dụng cho datasets tương tự**, với các bước kiểm chứng và sửa lỗi rõ ràng
3. **Cung cấp foundation cho advanced customer analytics**, bao gồm cả khả năng diễn giải (SHAP) — không chỉ là một "hộp đen"
4. **Triển khai thành ứng dụng thực tế** (Streamlit Dashboard) có thể cập nhật dữ liệu và xuất báo cáo, thu hẹp khoảng cách giữa nghiên cứu học thuật và vận hành kinh doanh

Đây là một example điển hình của việc áp dụng machine learning để giải quyết real-world business problems, từ data understanding, model development có kiểm chứng nghiêm ngặt, đến diễn giải kết quả và triển khai thực tế.