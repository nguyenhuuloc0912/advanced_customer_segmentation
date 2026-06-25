# -*- coding: utf-8 -*-
"""
Thư viện phân khúc khách hàng (Customer Segmentation Library).

File này gom các class phục vụ toàn bộ quy trình phân khúc khách hàng:
1. DataCleaner: đọc dữ liệu, làm sạch dữ liệu giao dịch và tính RFM cơ bản.
2. FeatureEngineer: tạo đặc trưng ở cấp độ khách hàng từ dữ liệu giao dịch.
3. ClusterAnalyzer: PCA, chọn số cụm, K-Means, radar chart và giải thích bằng SHAP.
4. DataVisualizer: vẽ các biểu đồ EDA phục vụ phân tích hành vi khách hàng.

Lưu ý: các comment tiếng Việt được thêm để giúp dễ đọc code, không thay đổi logic xử lý.
"""

# Import các thư viện xử lý thời gian, đường dẫn file và thư mục
import datetime as dt
import os

# Import các thư viện phân tích dữ liệu, trực quan hóa và giải thích mô hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
# Import các công cụ thống kê, đặc biệt Box-Cox để biến đổi phân phối dữ liệu
from scipy import stats
from scipy.stats import boxcox
# Import các thuật toán và thước đo từ scikit-learn phục vụ PCA, clustering và đánh giá mô hình
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

class DataCleaner:
    """
    Class phụ trách đọc, làm sạch và tiền xử lý dữ liệu giao dịch bán lẻ.
    
    Ý nghĩa chính:
    - Load dữ liệu thô từ file CSV.
    - Loại bỏ giao dịch không hợp lệ như hóa đơn hủy, quantity/price âm hoặc bằng 0.
    - Lọc dữ liệu khách hàng tại United Kingdom.
    - Tạo dữ liệu RFM dùng cho phân tích hành vi khách hàng.
    
    """

    def __init__(self, data_path):
        """
        Khởi tạo đối tượng DataCleaner.
        
        Args:
        data_path (str): Đường dẫn đến file dữ liệu thô.
        
        """
        self.data_path = data_path
        self.df = None
        self.df_uk = None
        self.rfm_data = None

    def load_data(self):
        """
        Đọc dữ liệu thô từ file CSV và chuẩn hóa kiểu dữ liệu ban đầu.
        
        Returns:
        pd.DataFrame: DataFrame chứa dữ liệu vừa đọc từ file.
        
        """
        # Khai báo kiểu dữ liệu rõ ràng để đọc CSV ổn định hơn, tránh pandas tự suy đoán sai kiểu.
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )

        # Đọc dữ liệu từ CSV; parse_dates giúp InvoiceDate được chuyển sẵn sang kiểu datetime.
        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        # Chuyển CustomerID thành format 6 ký tự
        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )

        print(f"Kích thước dữ liệu: {self.df.shape}")
        print(f"Số bản ghi: {len(self.df):,}")

        return self.df

    def clean_data(self):
        """
        Làm sạch dữ liệu giao dịch.
        
        Các bước chính:
        - Tính TotalPrice = Quantity * UnitPrice.
        - Loại bỏ hóa đơn bị hủy.
        - Chỉ giữ khách hàng ở United Kingdom.
        - Loại bỏ bản ghi thiếu CustomerID hoặc có Quantity/UnitPrice không hợp lệ.
        
        Returns:
        pd.DataFrame: Dữ liệu UK đã được làm sạch.
        
        """
        # Thêm cột TotalPrice
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Loại bỏ các hóa đơn bị hủy (bắt đầu bằng 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # Chỉ tập trung vào khách hàng UK
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Loại bỏ bản ghi thiếu CustomerID
        self.df_uk = self.df_uk.dropna(subset=["CustomerID"])
        
        # self.df_uk = self.df_uk[
        #     ~self.df_uk["CustomerID"].str.contains("nan", case=False, na=False)
        # ]

        # Loại bỏ các sản phẩm có quantity hoặc price không hợp lệ
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]

        return self.df_uk

    def create_time_features(self):
        """
        Tạo thêm các đặc trưng thời gian từ InvoiceDate.
        
        DayOfWeek giúp biết khách thường mua vào thứ mấy.
        HourOfDay giúp biết khách thường mua vào khung giờ nào.
        
        """
        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def calculate_rfm(self):
        """
        Tính các chỉ số RFM cho từng khách hàng.
        
        RFM gồm:
        - Recency: số ngày kể từ lần mua gần nhất.
        - Frequency: số hóa đơn/giao dịch duy nhất.
        - Monetary: tổng số tiền khách hàng đã chi tiêu.
        
        Returns:
        pd.DataFrame: Bảng RFM theo từng CustomerID.
        
        """
        # Chọn mốc thời gian giả định là sau ngày giao dịch cuối cùng 1 ngày để tính Recency.
        snapshot_date = self.df_uk["InvoiceDate"].max() + pd.Timedelta(days=1)

        self.rfm_data = self.df_uk.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": lambda x: len(x.unique()),  # Frequency
                "TotalPrice": lambda x: x.sum(),  # Monetary
            }
        )

        self.rfm_data.columns = ["Recency", "Frequency", "Monetary"]
        return self.rfm_data

    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Lưu dữ liệu đã làm sạch ra thư mục processed.
        
        Args:
        output_dir (str): Thư mục đầu ra để lưu file CSV.
        
        """
        # Tạo thư mục nếu chưa tồn tại, sau đó lưu dữ liệu sạch ra file CSV.
        os.makedirs(output_dir, exist_ok=True)
        self.df_uk.to_csv(f"{output_dir}/cleaned_uk_data.csv", index=False)
        print(f"Đã lưu dữ liệu đã làm sạch: {output_dir}/cleaned_uk_data.csv")


class FeatureEngineer:
    """
    Class phụ trách tạo đặc trưng ở cấp độ khách hàng.
    
    Dữ liệu ban đầu nằm ở cấp độ giao dịch/sản phẩm. Class này tổng hợp dữ liệu
    thành một dòng cho mỗi CustomerID, sau đó biến đổi và chuẩn hóa đặc trưng
    để đưa vào mô hình clustering.
    
    """

    def __init__(self, data_path):
        """
        Khởi tạo FeatureEngineer với đường dẫn dữ liệu đã làm sạch.
        
        Args:
        data_path (str): Đường dẫn đến file cleaned_uk_data.csv.
        
        """
        self.data_path = data_path
        self.df = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.customer_features_scaled = None

        # Danh sách các feature sẽ được tính cho từng khách hàng.
        # Các feature này mô tả quy mô mua hàng, tần suất mua, giá trị giao dịch và mức độ đa dạng sản phẩm.
        self.feature_customer = [
            "Sum_Quantity",
            "Mean_UnitPrice",
            "Mean_TotalPrice",
            "Sum_TotalPrice",
            "Count_Invoice",
            "Count_Stock",
            "Mean_InvoiceCountPerStock",
            "Mean_StockCountPerInvoice",
            "Mean_UnitPriceMeanPerInvoice",
            "Mean_QuantitySumPerInvoice",
            "Mean_TotalPriceMeanPerInvoice",
            "Mean_TotalPriceSumPerInvoice",
            "Mean_UnitPriceMeanPerStock",
            "Mean_QuantitySumPerStock",
            "Mean_TotalPriceMeanPerStock",
            "Mean_TotalPriceSumPerStock",
        ]

        self.feature_customer2 = ["CustomerID"] + self.feature_customer

    def load_data(self):
        """
        Đọc dữ liệu đã làm sạch và chuyển InvoiceDate về kiểu datetime.
        
        Returns:
        pd.DataFrame: DataFrame dữ liệu giao dịch đã làm sạch.
        
        """
        self.df = pd.read_csv(self.data_path)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])

        print(f"Kích thước dữ liệu: {self.df.shape}")
        return self.df

    def create_customer_features(self):
        """
        Tạo bộ đặc trưng tổng hợp cho từng khách hàng.
        
        Mỗi khách hàng sẽ được biểu diễn bằng các chỉ số như tổng số lượng mua,
        tổng chi tiêu, số hóa đơn, số loại sản phẩm đã mua, giá trị trung bình
        trên mỗi hóa đơn/sản phẩm, v.v.
        
        Returns:
        pd.DataFrame: Bảng đặc trưng theo từng CustomerID.
        
        """
        num_customers = self.df["CustomerID"].nunique()
        # Tạo DataFrame rỗng ban đầu: mỗi dòng tương ứng 1 khách hàng, mỗi cột là 1 feature.
        self.customer_features = pd.DataFrame(
            data=np.zeros((num_customers, len(self.feature_customer2)), dtype=float),
            columns=self.feature_customer2,
        )

        self.customer_features["CustomerID"] = self.customer_features[
            "CustomerID"
        ].astype("object")

        print("Đang tính toán features cho từng khách hàng...")

        # Gom dữ liệu theo CustomerID để tính feature riêng cho từng khách hàng.
        for i, (customer_id, value) in enumerate(self.df.groupby("CustomerID")):
            # Gán mã khách hàng vào dòng hiện tại
            self.customer_features.iat[i, 0] = customer_id

            # 1. Tổng quantity
            self.customer_features.iat[i, 1] = value.Quantity.sum()

            # 2. Giá trung bình
            self.customer_features.iat[i, 2] = value.UnitPrice.mean()

            # 3. Giá trị giao dịch trung bình
            self.customer_features.iat[i, 3] = value.TotalPrice.mean()

            # 4. Tổng chi tiêu
            self.customer_features.iat[i, 4] = value.TotalPrice.sum()

            # 5. Số hóa đơn
            self.customer_features.iat[i, 5] = value.InvoiceNo.nunique()

            # 6. Số loại sản phẩm
            self.customer_features.iat[i, 6] = value.StockCode.nunique()

            # 7-16. Các chỉ số hành vi nâng cao theo sản phẩm và theo hóa đơn.
            # Những feature này giúp mô hình hiểu sâu hơn về cách khách mua hàng, không chỉ tổng chi tiêu.
            self.customer_features.iat[i, 7] = value.groupby("StockCode").size().mean()
            self.customer_features.iat[i, 8] = value.groupby("InvoiceNo").size().mean()
            self.customer_features.iat[i, 9] = (
                value.groupby("InvoiceNo")["UnitPrice"].mean().mean()
            )
            self.customer_features.iat[i, 10] = (
                value.groupby("InvoiceNo")["Quantity"].sum().mean()
            )
            self.customer_features.iat[i, 11] = (
                value.groupby("InvoiceNo")["TotalPrice"].mean().mean()
            )
            self.customer_features.iat[i, 12] = (
                value.groupby("InvoiceNo")["TotalPrice"].sum().mean()
            )
            self.customer_features.iat[i, 13] = (
                value.groupby("StockCode")["UnitPrice"].mean().mean()
            )
            self.customer_features.iat[i, 14] = (
                value.groupby("StockCode")["Quantity"].sum().mean()
            )
            self.customer_features.iat[i, 15] = (
                value.groupby("StockCode")["TotalPrice"].mean().mean()
            )
            self.customer_features.iat[i, 16] = (
                value.groupby("StockCode")["TotalPrice"].sum().mean()
            )

            if (i + 1) % 500 == 0:
                print(f"Đã xử lý {i + 1}/{num_customers} khách hàng...")

        print("✓ Hoàn thành tính toán features!")
        return self.customer_features

    def transform_features(self):
        """
        Áp dụng Box-Cox transformation để làm phân phối dữ liệu bớt lệch.
        
        Box-Cox thường dùng khi dữ liệu numeric bị lệch phải mạnh, ví dụ tổng
        chi tiêu của một vài khách hàng rất cao so với phần còn lại.
        
        Returns:
        pd.DataFrame: Bộ đặc trưng sau khi biến đổi Box-Cox.
        
        """
        # Đặt CustomerID làm index vì CustomerID là mã định danh, không nên đưa vào tính toán numeric.
        customer_features_indexed = self.customer_features.set_index("CustomerID")

        # Áp dụng Box-Cox để giảm skewness; cộng 1 để đảm bảo dữ liệu dương.
        feature_values = customer_features_indexed.values + 1  # Cộng 1 cho Box-Cox

        self.customer_features_transformed = customer_features_indexed.copy()

        print("Đang áp dụng Box-Cox transformation...")
        for i, feature in enumerate(self.feature_customer):
            transformed, lambda_param = boxcox(feature_values[:, i])
            self.customer_features_transformed.iloc[:, i] = transformed

        print("✓ Box-Cox transformation hoàn thành!")
        return self.customer_features_transformed

    def scale_features(self):
        """
        Chuẩn hóa dữ liệu bằng StandardScaler.
        
        Sau bước này, mỗi feature thường có trung bình gần 0 và độ lệch chuẩn gần 1.
        Điều này rất quan trọng với K-Means vì thuật toán dựa trên khoảng cách.
        
        Returns:
        pd.DataFrame: Bộ đặc trưng đã được scale.
        
        """
        # StandardScaler đưa các feature về cùng thang đo, tránh feature có giá trị lớn chi phối K-Means.
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.customer_features_transformed)

        self.customer_features_scaled = pd.DataFrame(
            features_scaled,
            columns=self.feature_customer,
            index=self.customer_features_transformed.index,
        )

        print("✓ Feature scaling hoàn thành!")
        return self.customer_features_scaled

    def plot_features_boxplots(self, transformed=False, save_path=None):
        """
        Vẽ boxplot cho toàn bộ features để quan sát outlier và độ phân tán.
        
        Args:
        transformed (bool): True nếu muốn vẽ dữ liệu sau Box-Cox, False nếu vẽ dữ liệu gốc.
        save_path (str): Đường dẫn lưu ảnh nếu muốn xuất biểu đồ ra file.
        
        """
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("CustomerID")
                title = "Box Plots của Features Gốc (Trước Box-Cox Transformation)"
            else:
                print(
                    "Lỗi: Chưa có dữ liệu features. Hãy chạy create_customer_features() trước."
                )
                return

        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.boxplot(data.iloc[:, i] if transformed else data[feature])
                plt.title(feature, fontsize=10)
                plt.xticks([])

            plt.tight_layout()
            # plt.suptitle(title, fontsize=16, y=1.1)

            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Đã lưu biểu đồ: {save_path}")

            plt.show()

    def plot_features_histograms(self, transformed=False, save_path=None):
        """
        Vẽ histogram cho toàn bộ features để quan sát phân phối dữ liệu.
        
        Args:
        transformed (bool): True nếu muốn vẽ dữ liệu sau Box-Cox, False nếu vẽ dữ liệu gốc.
        save_path (str): Đường dẫn lưu ảnh nếu muốn xuất biểu đồ ra file.
        
        """
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
            title = "Histograms của Features sau Box-Cox Transformation"
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("CustomerID")
                title = "Histograms của Features Gốc (Trước Box-Cox Transformation)"
            else:
                print(
                    "Lỗi: Chưa có dữ liệu features. Hãy chạy create_customer_features() trước."
                )
                return

        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.hist(
                    data.iloc[:, i] if transformed else data[feature],
                    bins=30,
                    alpha=0.9,
                )
                plt.title(feature, fontsize=10)
                plt.ylabel("Tần suất", fontsize=8)

            plt.tight_layout()
            # plt.suptitle(title, fontsize=16, y=0.98)

            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Đã lưu biểu đồ: {save_path}")

            plt.show()

    def save_features(self, output_dir="../data/processed"):
        """
        Lưu các bộ feature đã xử lý ra file CSV.
        
        Bao gồm:
        - customer_features.csv: feature gốc.
        - customer_features_transformed.csv: feature sau Box-Cox.
        - customer_features_scaled.csv: feature sau chuẩn hóa.
        
        Args:
        output_dir (str): Thư mục đầu ra để lưu file.
        
        """
        os.makedirs(output_dir, exist_ok=True)

        # Lưu feature gốc để dùng cho việc giải thích business sau khi phân cụm.
        customer_features_indexed = self.customer_features.set_index("CustomerID")
        customer_features_indexed.to_csv(f"{output_dir}/customer_features.csv")

        # Lưu feature đã biến đổi Box-Cox để kiểm tra/phân tích lại khi cần.
        self.customer_features_transformed.to_csv(
            f"{output_dir}/customer_features_transformed.csv"
        )

        # Lưu feature đã scale; đây thường là input chính cho K-Means/PCA.
        self.customer_features_scaled.to_csv(
            f"{output_dir}/customer_features_scaled.csv"
        )

        print(f"✓ Đã lưu tất cả features vào: {output_dir}")


class ClusterAnalyzer:
    """
    Class phụ trách phân tích clustering và trực quan hóa kết quả phân cụm.
    
    Các chức năng chính:
    - Đọc feature gốc và feature đã chuẩn hóa.
    - Giảm chiều dữ liệu bằng PCA.
    - Tìm số cụm phù hợp bằng Elbow và Silhouette Score.
    - Huấn luyện K-Means với nhiều giá trị k.
    - Vẽ biểu đồ PCA, radar chart và giải thích cụm bằng SHAP.
    
    """

    # Mapping tên feature tiếng Anh sang tiếng Việt để biểu đồ/báo cáo dễ hiểu hơn.
    FEATURE_NAMES_VN = {
        "Sum_Quantity": "Tổng số lượng mua",
        "Mean_UnitPrice": "Giá trung bình",
        "Mean_TotalPrice": "Giá trị giao dịch TB",
        "Sum_TotalPrice": "Tổng chi tiêu",
        "Count_Invoice": "Số lần mua",
        "Count_Stock": "Số sản phẩm khác nhau",
        "Mean_InvoiceCountPerStock": "Tần suất mua/sản phẩm",
        "Mean_StockCountPerInvoice": "Sản phẩm/giao dịch",
        "Mean_UnitPriceMeanPerInvoice": "Giá TB/giao dịch",
        "Mean_QuantitySumPerInvoice": "Số lượng/giao dịch",
        "Mean_TotalPriceMeanPerInvoice": "Giá trị TB/giao dịch",
        "Mean_TotalPriceSumPerInvoice": "Tổng giá trị/giao dịch",
        "Mean_UnitPriceMeanPerStock": "Giá TB/sản phẩm",
        "Mean_QuantitySumPerStock": "Số lượng TB/sản phẩm",
        "Mean_TotalPriceMeanPerStock": "Giá trị TB/sản phẩm",
        "Mean_TotalPriceSumPerStock": "Tổng giá trị/sản phẩm",
    }

    def __init__(self, scaled_features_path, original_features_path):
        """
        Khởi tạo ClusterAnalyzer với đường dẫn feature đã scale và feature gốc.
        
        Args:
        scaled_features_path (str): Đường dẫn file customer_features_scaled.csv.
        original_features_path (str): Đường dẫn file customer_features.csv.
        
        """
        self.scaled_features_path = scaled_features_path
        self.original_features_path = original_features_path
        self.df_scaled = None
        self.df_original = None
        self.df_pca = None
        self.pca = None
        self.optimal_clusters = {}
        self.cluster_results = {}
        self.surrogate_models = {}
        self.shap_results = {}

    def load_data(self):
        """
        Đọc dữ liệu feature đã scale và feature gốc.
        
        Returns:
        tuple: Gồm (df_scaled, df_original).
        
        """
        self.df_scaled = pd.read_csv(self.scaled_features_path, index_col=0)
        self.df_original = pd.read_csv(self.original_features_path, index_col=0)

        print(f"Số khách hàng: {self.df_scaled.shape[0]}")
        print(f"Số features: {self.df_scaled.shape[1]}")

        return self.df_scaled, self.df_original

    def apply_pca(self, n_components=None):
        """
        Áp dụng PCA để giảm chiều dữ liệu.
        
        PCA giúp nén nhiều features về một số trục chính, thường dùng để:
        - Vẽ dữ liệu 2D/3D dễ quan sát hơn.
        - Xem các chiều chính giải thích được bao nhiêu phương sai.
        
        Args:
        n_components (int): Số thành phần chính muốn giữ lại. Nếu None thì giữ tối đa.
        
        Returns:
        pd.DataFrame: Dữ liệu sau khi biến đổi PCA.
        
        """
        # PCA học các trục chính từ dữ liệu đã scale rồi chiếu dữ liệu sang không gian mới.
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(self.df_scaled)

        pca_columns = [f"PC{i+1}" for i in range(pca_features.shape[1])]
        self.df_pca = pd.DataFrame(
            pca_features, columns=pca_columns, index=self.df_scaled.index
        )

        print(f"PCA shape: {self.df_pca.shape}")
        return self.df_pca

    def plot_pca_variance(self):
        """
        Vẽ biểu đồ tỷ lệ phương sai được giải thích bởi từng thành phần PCA.
        
        Biểu đồ này giúp quyết định nên giữ bao nhiêu principal components
        để vẫn giữ được phần lớn thông tin trong dữ liệu.
        
        """
        plt.figure(figsize=(12, 6))

        plt.bar(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            self.pca.explained_variance_ratio_,
            alpha=0.7,
            label="Phương sai riêng lẻ",
        )

        plt.step(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            np.cumsum(self.pca.explained_variance_ratio_),
            where="mid",
            label="Phương sai tích lũy",
            color="red",
            linewidth=2,
        )

        plt.axhline(y=0.8, color="green", linestyle="--", label="80% phương sai")
        plt.axhline(y=0.9, color="orange", linestyle="--", label="90% phương sai")

        plt.xlabel("Thành phần chính")
        plt.ylabel("Tỷ lệ phương sai được giải thích")
        plt.title("Phân tích PCA - Phương sai được giải thích")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\nPhương sai tích lũy:")
        for i in range(min(5, len(self.pca.explained_variance_ratio_))):
            cumsum = np.sum(self.pca.explained_variance_ratio_[: i + 1])
            print(f"PC1-PC{i+1}: {cumsum:.2%}")

    def find_optimal_clusters(self, k_range=range(2, 11)):
        """
        Tìm số lượng cụm phù hợp bằng Elbow Method và Silhouette Score.
        
        Args:
        k_range (range): Dải giá trị k cần thử, ví dụ range(2, 11).
        
        Returns:
        dict: Kết quả inertia, silhouette score và k tốt nhất theo silhouette.
        
        """
        # inertia đo tổng khoảng cách trong cụm; silhouette đo mức độ tách biệt giữa các cụm.
        inertias = []
        silhouette_scores = []

        # Thử từng giá trị k để so sánh chất lượng phân cụm.
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.df_scaled, labels))

        self.optimal_clusters = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "best_k_silhouette": list(k_range)[np.argmax(silhouette_scores)],
        }

        return self.optimal_clusters

    def plot_optimal_clusters(self):
        """
        Vẽ biểu đồ Elbow và Silhouette Score để hỗ trợ chọn số cụm.
        
        Elbow quan sát điểm mà inertia giảm chậm lại.
        Silhouette Score đo mức độ tách biệt giữa các cụm, càng cao thường càng tốt.
        
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Elbow Method: tìm điểm gãy khi tăng k không còn giúp giảm inertia nhiều nữa.
        axes[0].plot(
            self.optimal_clusters["k_range"],
            self.optimal_clusters["inertias"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="blue",
        )
        axes[0].set_xlabel("Số lượng clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Phương pháp Elbow")
        axes[0].grid(True, alpha=0.3)

        # Silhouette Score: càng cao thì các cụm càng tách biệt và ít chồng lấn hơn.
        axes[1].plot(
            self.optimal_clusters["k_range"],
            self.optimal_clusters["silhouette_scores"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="green",
        )
        axes[1].set_xlabel("Số lượng clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Phương pháp Silhouette Score")
        axes[1].grid(True, alpha=0.3)

        best_k = self.optimal_clusters["best_k_silhouette"]
        best_score = max(self.optimal_clusters["silhouette_scores"])
        axes[1].scatter(best_k, best_score, s=200, c="red", alpha=0.5, zorder=5)
        axes[1].annotate(
            f"Tốt nhất k={best_k}",
            xy=(best_k, best_score),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        plt.tight_layout()
        plt.show()

        print(f"Silhouette Score đề xuất: k={best_k} (điểm số = {best_score:.3f})")

    def apply_kmeans(self, k_values=[3, 4]):
        """
        Huấn luyện K-Means với một hoặc nhiều giá trị k.
        
        Args:
        k_values (list): Danh sách số cụm muốn thử, ví dụ [3, 4].
        
        Returns:
        dict: Kết quả phân cụm, kích thước cụm và trung bình feature theo cụm.
        
        """
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.df_scaled)

            # Thêm nhãn cụm vào cả dữ liệu scaled, PCA và original để tiện vẽ biểu đồ/giải thích.
            cluster_col = f"Cluster_{k}"
            self.df_scaled[cluster_col] = clusters
            self.df_pca[cluster_col] = clusters
            self.df_original[cluster_col] = clusters

            self.cluster_results[k] = {
                "labels": clusters,
                "sizes": pd.Series(clusters).value_counts().sort_index(),
                "means": self.df_original.groupby(cluster_col).mean(),
            }

            print(f"Kích thước clusters (k={k}):")
            print(self.cluster_results[k]["sizes"])

        return self.cluster_results

    def plot_clusters_pca(self, k_values=[3, 4]):
        """
        Vẽ kết quả phân cụm trên mặt phẳng PCA 2D.
        
        Args:
        k_values (list): Danh sách các giá trị k cần trực quan hóa.
        
        """
        fig, axes = plt.subplots(1, len(k_values), figsize=(16, 6))
        if len(k_values) == 1:
            axes = [axes]

        for i, k in enumerate(k_values):
            cluster_col = f"Cluster_{k}"
            scatter = axes[i].scatter(
                self.df_pca["PC1"],
                self.df_pca["PC2"],
                c=self.df_pca[cluster_col],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_title(f"Phân cụm K-Means (k={k})")
            plt.colorbar(scatter, ax=axes[i], label="Cluster")

        plt.tight_layout()
        plt.show()

    def plot_clusters_pca_3d(self, k_values=[3, 4]):
        """
        Vẽ kết quả phân cụm trong không gian PCA 3D.
        
        Args:
        k_values (list): Danh sách các giá trị k cần trực quan hóa.
        
        """
        # Import 3D plotting ngay trong hàm vì chỉ hàm này cần dùng.
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 6))

        for i, k in enumerate(k_values):
            cluster_col = f"Cluster_{k}"
            ax = fig.add_subplot(1, len(k_values), i + 1, projection="3d")

            scatter = ax.scatter(
                self.df_pca["PC1"],
                self.df_pca["PC2"],
                self.df_pca["PC3"],
                c=self.df_pca[cluster_col],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f"Phân cụm K-Means 3D (k={k})")

            # Thêm thanh màu để biết mỗi màu tương ứng cluster nào.
            plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5)

        plt.tight_layout()
        plt.show()

    def create_radar_chart(self, k, cluster_names=None):
        """
        Tạo radar chart tổng hợp để so sánh đặc điểm các cụm khách hàng.
        
        Args:
        k (int): Số cụm đang phân tích.
        cluster_names (list): Tên tùy chỉnh cho từng cụm, nếu có.
        
        """
        cluster_means = self.cluster_results[k]["means"]

        # Chọn features quan trọng cho radar chart
        important_features = {
            "Sum_Quantity": "Khối lượng mua",
            "Sum_TotalPrice": "Tổng chi tiêu",
            "Mean_UnitPrice": "Mức giá ưa thích",
            "Count_Invoice": "Tần suất mua",
            "Count_Stock": "Đa dạng sản phẩm",
            "Mean_TotalPriceSumPerInvoice": "Giá trị/giao dịch",
        }

        # Lọc các feature quan trọng và chuẩn hóa dữ liệu
        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        # Chuẩn hóa global theo min-max để các feature có thể so sánh trên cùng radar chart.
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Thay thế bằng labels tiếng Việt
        data_normalized.columns = [
            important_features[col] for col in data_normalized.columns
        ]

        # Thiết lập các góc trên biểu đồ radar, mỗi góc đại diện cho một feature.
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Chọn màu cho từng cluster để biểu đồ dễ phân biệt.
        colors = (
            ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
            if k == 4
            else ["#E74C3C", "#2ECC71", "#3498DB"]
        )
        if not cluster_names:
            cluster_names = [f"Nhóm {i}" for i in range(k)]

        # Tạo biểu đồ radar dạng polar plot.
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = (
                cluster_names[idx] if idx < len(cluster_names) else f"Nhóm {idx}"
            )

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=4,
                label=cluster_name,
                color=color,
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.15, color=color)

        # Tinh chỉnh giao diện biểu đồ cho dễ đọc và chuyên nghiệp hơn.
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, weight="bold", color="#2C3E50")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["20%", "40%", "60%", "80%", "100%"], size=10, color="#7F8C8D"
        )
        ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
        ax.set_facecolor("#FAFAFA")

        ax.set_title(
            f"Phân tích phân khúc khách hàng (K={k})",
            size=16,
            weight="bold",
            pad=30,
            color="#2C3E50",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

        plt.tight_layout()
        plt.show()

    def create_individual_radar_plots(self, k, cluster_names=None):
        """
        Tạo radar chart riêng cho từng cụm để xem profile chi tiết hơn.
        
        Args:
        k (int): Số cụm đang phân tích.
        cluster_names (list): Tên tùy chỉnh cho từng cụm, nếu có.
        
        """
        cluster_means = self.cluster_results[k]["means"]

        # Chọn features quan trọng
        important_features = {
            "Sum_Quantity": "Khối lượng mua",
            "Sum_TotalPrice": "Tổng chi tiêu",
            "Mean_UnitPrice": "Mức giá ưa thích",
            "Count_Invoice": "Tần suất mua",
            "Count_Stock": "Đa dạng sản phẩm",
            "Mean_TotalPriceSumPerInvoice": "Giá trị/giao dịch",
            "Mean_TotalPriceMeanPerStock": "Chi tiêu/sản phẩm",
            "Mean_StockCountPerInvoice": "Sản phẩm/giao dịch",
        }

        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        # Chuẩn hóa dữ liệu
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Thay thế labels tiếng Việt
        data_normalized.columns = [
            important_features[col] for col in data_normalized.columns
        ]

        # Tính góc cho từng feature trên radar plot.
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Colors chuyên nghiệp
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        if not cluster_names:
            cluster_names = [f"Nhóm {i}" for i in range(k)]

        # Tạo subplot cho từng cluster với layout tối ưu
        if k == 4:
            # Layout 2x2 cho k=4
            nrows, ncols = 2, 2
            figsize = (12, 10)
        else:
            # Layout 1 hàng cho các trường hợp khác
            nrows, ncols = 1, k
            figsize = (5 * k, 5)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, subplot_kw=dict(projection="polar")
        )

        # Đảm bảo axes luôn là array 2D để dễ xử lý
        if k == 1:
            axes = np.array([[axes]])
        elif k == 4:
            # axes đã là 2D array (2x2)
            pass
        else:
            # Chuyển thành 2D array cho consistency
            axes = axes.reshape(1, -1)

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            # Tính toán vị trí trong grid 2D
            if k == 4:
                row_idx, col_idx = idx // 2, idx % 2
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[0, idx] if len(axes.shape) == 2 else axes[idx]

            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = (
                cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"
            )

            # Vẽ đường radar và vùng tô màu cho cluster hiện tại
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=3,
                label=cluster_name,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.25, color=color)

            # Styling chuyên nghiệp
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=11, weight="bold", color="#2C3E50")
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(
                ["20%", "40%", "60%", "80%", "100%"], size=9, color="#7F8C8D"
            )
            ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
            ax.set_facecolor("#FAFAFA")

            # Tiêu đề cho từng biểu đồ con
            ax.set_title(
                f"{cluster_name}\n({cluster_means.index[idx]})",
                size=13,
                weight="bold",
                pad=20,
                color=color,
            )

        plt.suptitle(
            f"Phân tích chi tiết từng Cluster (K={k})", size=16, weight="bold", y=1.05
        )
        plt.tight_layout()
        plt.show()

    # def train_surrogate_model(self, k):
    #     """
    #     Huấn luyện mô hình RandomForest để mô phỏng kết quả phân cụm của K-Means.
        
    #     Vì K-Means khó giải thích trực tiếp bằng SHAP, ta dùng RandomForest học lại
    #     nhãn cụm của K-Means. Nếu RandomForest dự đoán lại nhãn cụm đủ tốt, ta có
    #     thể dùng SHAP trên RandomForest để hiểu feature nào ảnh hưởng đến từng cụm.
        
    #     Args:
    #     k (int): Số cụm cần giải thích.
        
    #     Returns:
    #     dict: Mô hình thay thế và các chỉ số đánh giá như accuracy, confusion matrix.
        
    #     """
    #     if k not in self.cluster_results:
    #         raise ValueError(f"Cluster results for k={k} not found. Run apply_kmeans first.")
        
    #     # Lấy tất cả cột không phải là Cluster_
    #     feature_cols = [col for col in self.df_scaled.columns if not col.startswith('Cluster_')]
    #     X = self.df_scaled[feature_cols].values
    #     y = self.cluster_results[k]['labels']
        
    #     # Huấn luyện mô hình RandomForest classifier
    #     rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    #     rf_model.fit(X, y)
        
    #     # Dự đoán lại nhãn cụm bằng mô hình thay thế
    #     y_pred = rf_model.predict(X)
        
    #     # Tính các chỉ số để xem RandomForest mô phỏng K-Means tốt đến đâu
    #     accuracy = accuracy_score(y, y_pred)
    #     conf_matrix = confusion_matrix(y, y_pred)
        
    #     # Lưu model, metric và thông tin feature để dùng ở bước SHAP
    #     self.surrogate_models[k] = {
    #         'model': rf_model,
    #         'accuracy': accuracy,
    #         'confusion_matrix': conf_matrix,
    #         'feature_names': feature_cols
    #     }
        
    #     # In báo cáo kết quả huấn luyện mô hình thay thế
    #     print(f"=== HUẤN LUYỆN MÔ HÌNH THAY THẾ (k={k}) ===")
    #     print(f"Độ chính xác: {accuracy:.4f} ({accuracy*100:.2f}%)")
    #     print(f"\nConfusion Matrix:")
    #     print(conf_matrix)
    #     print(f"\nMô hình có thể dự đoán {'CHÍNH XÁC' if accuracy >= 0.95 else 'HỢP LÝ'} các phân cụm.")
        
    #     return self.surrogate_models[k]
    def train_surrogate_model(self, k, test_size=0.2):
        if k not in self.cluster_results:
            raise ValueError(f"Cluster results for k={k} not found.")

        feature_cols = [col for col in self.df_scaled.columns 
                        if not col.startswith('Cluster_')]
        X = self.df_scaled[feature_cols].values
        y = self.cluster_results[k]['labels']

        # ✅ Split trước khi train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # ✅ Giới hạn depth để tránh memorize
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,        # thêm dòng này
            random_state=42,
            n_jobs=-1
        )

        # ✅ Cross-validation để đánh giá robust hơn
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
        
        # Train final model trên toàn bộ X để SHAP dùng full data
        rf_model.fit(X, y)

        # ✅ Evaluate trên test set
        y_pred_test = rf_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_accuracy = accuracy_score(y, rf_model.predict(X))

        conf_matrix = confusion_matrix(y_test, y_pred_test)

        self.surrogate_models[k] = {
            'model': rf_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'confusion_matrix': conf_matrix,
            'feature_names': feature_cols
        }

        print(f"=== SURROGATE MODEL (k={k}) ===")
        print(f"Train accuracy : {train_accuracy*100:.2f}%")
        print(f"Test accuracy  : {test_accuracy*100:.2f}%")   # ← con số này mới quan trọng
        print(f"CV scores      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\nConfusion Matrix (test set):")
        print(conf_matrix)

        # ✅ Cảnh báo nếu overfit
        gap = train_accuracy - test_accuracy
        if gap > 0.05:
            print(f"\n⚠️  Overfit gap: {gap:.2%} — cân nhắc tăng max_depth hoặc min_samples_leaf")
        else:
            print(f"\n✅ Model ổn định (gap = {gap:.2%})")

        return self.surrogate_models[k]

    def calculate_shap_values(self, k):
        """
        Tính SHAP values để giải thích lý do khách hàng thuộc từng cụm.
        
        Args:
        k (int): Số cụm cần phân tích SHAP.
        
        Returns:
        dict: SHAP explainer, SHAP values, dữ liệu đầu vào và tên features.
        
        """
        if k not in self.surrogate_models:
            raise ValueError(f"Mô hình thay thế cho k={k} không tìm thấy. Vui lòng chạy train_surrogate_model trước.")
        
        # Lấy mô hình thay thế và danh sách feature đã dùng khi huấn luyện
        rf_model = self.surrogate_models[k]['model']
        feature_cols = self.surrogate_models[k]['feature_names']
        X = self.df_scaled[feature_cols].values
        
        # Tạo SHAP explainer để tính mức đóng góp của từng feature
        # Khi dữ liệu làm nền càng lớn thì thuật toán SHAP càng chính xác
        print(f"Tính toán SHAP values cho {len(X):,} khách hàng...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values_raw = explainer.shap_values(X)
        
        # Chuyển SHAP values sang dạng list cho bài toán nhiều cluster
        # Shape: (n_samples, n_features, n_classes) -> list (n_samples, n_features)
        if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            # Trường hợp nhiều lớp: tách SHAP values theo từng cluster
            shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
        else:
            # Trường hợp nhị phân hoặc SHAP đã trả về đúng định dạng
            shap_values = shap_values_raw
        
        # Lưu model, metric và thông tin feature để dùng ở bước SHAP
        self.shap_results[k] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_cols,
            'X': X
        }
        
        print(f"Hoàn thành! SHAP values: {len(shap_values)} clusters, mỗi cluster shape: {shap_values[0].shape}")
        return self.shap_results[k]
    
    def plot_shap_summary(self, k, cluster_id=None):
        """
        Vẽ SHAP summary plot cho từng cụm.
        
        Biểu đồ này cho biết những feature nào quan trọng nhất trong việc dự đoán
        khách hàng thuộc vào một cluster cụ thể.
        
        Args:
        k (int): Số cụm đang phân tích.
        cluster_id (int, optional): Cluster cụ thể muốn vẽ. Hiện tại hàm đang lặp qua tất cả cluster.
        
        """
        if k not in self.shap_results:
            raise ValueError(f"Giá trị SHAP cho k={k} không tìm thấy. Vui lòng chạy calculate_shap_values trước.")
        
        shap_values = self.shap_results[k]['shap_values']
        X = self.shap_results[k]['X']
        feature_names = self.shap_results[k]['feature_names']
    
        for i in range(k):
            shap.summary_plot(
                shap_values[i],
                X,
                feature_names=feature_names,
                max_display=3,
                show=True
            )

    def save_clusters(self, output_dir="../data/processed"):
        """
        Lưu nhãn cụm của từng khách hàng ra file CSV.
        
        Args:
        output_dir (str): Thư mục đầu ra để lưu kết quả phân cụm.
        
        """
        os.makedirs(output_dir, exist_ok=True)

        for k in self.cluster_results.keys():
            cluster_col = f"Cluster_{k}"
            cluster_output = self.df_original[[cluster_col]].copy()
            cluster_output.columns = ["Cluster"]
            cluster_output = cluster_output.reset_index()
            cluster_output = cluster_output.sort_values(["Cluster", "CustomerID"])

            cluster_output.to_csv(
                f"{output_dir}/customer_clusters_k{k}.csv", index=False
            )
            print(
                f"Đã lưu kết quả phân cụm k={k}: {output_dir}/customer_clusters_k{k}.csv"
            )


class DataVisualizer:
    """
    Class phụ trách trực quan hóa dữ liệu và kết quả phân tích khách hàng.
    
    Các biểu đồ trong class này giúp hiểu doanh thu, thời điểm mua hàng,
    sản phẩm bán chạy, phân phối hành vi khách hàng và RFM.
    
    """

    def __init__(self):
        """
        Khởi tạo DataVisualizer và thiết lập style mặc định cho biểu đồ.
        
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Vẽ doanh thu theo ngày và theo tháng.
        
        Args:
        df (pd.DataFrame): DataFrame có cột InvoiceDate và TotalPrice.
        
        """
        # Doanh thu theo ngày: giúp xem xu hướng và biến động ngắn hạn.
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Doanh thu hàng ngày")
        plt.xlabel("Ngày")
        plt.ylabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

        # Doanh thu theo tháng: giúp xem xu hướng tổng quát theo thời gian.
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Doanh thu hàng tháng")
        plt.xlabel("Tháng")
        plt.ylabel("Doanh thu (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Vẽ heatmap thể hiện số lượng giao dịch theo thứ trong tuần và giờ trong ngày.
        
        Args:
        df (pd.DataFrame): DataFrame đã có các cột DayOfWeek và HourOfDay.
        
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.title("Hoạt động mua hàng theo ngày và giờ")
        plt.xlabel("Giờ trong ngày")
        plt.ylabel("Ngày trong tuần (0=Thứ 2, 6=Chủ nhật)")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df, top_n=10):
        """
        Vẽ top sản phẩm theo số lượng bán và theo doanh thu.
        
        Args:
        df (pd.DataFrame): DataFrame giao dịch.
        top_n (int): Số lượng sản phẩm top muốn hiển thị.
        
        """
        # Top sản phẩm theo số lượng
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} sản phẩm theo số lượng bán")
        plt.xlabel("Số lượng bán")
        plt.tight_layout()
        plt.show()

        # Top sản phẩm theo doanh thu
        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} sản phẩm theo doanh thu")
        plt.xlabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df):
        """
        Vẽ phân phối hành vi khách hàng.
        
        Bao gồm:
        - Số giao dịch trên mỗi khách hàng.
        - Tổng chi tiêu trên mỗi khách hàng, có lọc bớt outlier 1% cao nhất.
        
        Args:
        df (pd.DataFrame): DataFrame giao dịch.
        
        """
        # Phân phối số hóa đơn/giao dịch của từng khách hàng
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Phân phối số giao dịch trên mỗi khách hàng")
        plt.xlabel("Số giao dịch")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

        # Phân phối tổng chi tiêu của từng khách hàng
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Phân phối tổng chi tiêu trên mỗi khách hàng")
        plt.xlabel("Tổng chi tiêu (GBP)")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

    def plot_rfm_analysis(self, rfm_data):
        """
        Vẽ phân phối các chỉ số RFM.
        
        Args:
        rfm_data (pd.DataFrame): DataFrame chứa Recency, Frequency, Monetary.
        
        """
        # Vẽ lần lượt phân phối Recency, Frequency và Monetary để hiểu hành vi khách hàng.
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Phân phối Recency (Ngày kể từ lần mua cuối)")
        axes[0].set_xlabel("Ngày")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Phân phối Frequency (Số giao dịch)")
        axes[1].set_xlabel("Số giao dịch")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Phân phối Monetary (Tổng chi tiêu)")
        axes[2].set_xlabel("Tổng chi tiêu (GBP)")

        plt.tight_layout()
        plt.show()
