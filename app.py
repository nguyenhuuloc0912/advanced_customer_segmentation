"""
Customer Segmentation Dashboard
Streamlit app for interactive exploration of customer clustering results.

Expected files in data/processed/:
  - cleaned_uk_data.csv
  - customer_features.csv
  - customer_features_scaled.csv
  - customer_clusters_k3.csv
  - customer_clusters_k4.csv
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg: #0d0f1a;
    --surface: #151828;
    --card: #1c2035;
    --border: #2a2f4a;
    --accent1: #6c63ff;
    --accent2: #ff6584;
    --accent3: #43e97b;
    --accent4: #f7971e;
    --text: #e8eaf0;
    --muted: #8890a8;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
  }

  [data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: var(--accent1); }

  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
  }
  .metric-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  .cluster-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
  }

  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  .stSelectbox > div, .stSlider > div {
    background: var(--card) !important;
  }

  div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent1) !important;
    border-bottom: 2px solid var(--accent1);
  }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLUSTER_COLORS_K3 = {0: "#6c63ff", 1: "#ff6584", 2: "#43e97b"}
CLUSTER_COLORS_K4 = {0: "#6c63ff", 1: "#ff6584", 2: "#43e97b", 3: "#f7971e"}

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

RADAR_FEATURES = {
    "Sum_Quantity": "Khối lượng mua",
    "Sum_TotalPrice": "Tổng chi tiêu",
    "Mean_UnitPrice": "Mức giá ưa thích",
    "Count_Invoice": "Tần suất mua",
    "Count_Stock": "Đa dạng sản phẩm",
    "Mean_TotalPriceSumPerInvoice": "Giá trị/giao dịch",
    "Mean_TotalPriceMeanPerStock": "Chi tiêu/sản phẩm",
    "Mean_StockCountPerInvoice": "Sản phẩm/giao dịch",
}

SEGMENT_PERSONAS_K4 = {
    0: {"name": "💎 Khách VIP", "color": "#6c63ff", "desc": "Chi tiêu cao, mua thường xuyên, đa dạng sản phẩm"},
    1: {"name": "🔥 Khách Tiềm Năng", "color": "#ff6584", "desc": "Tần suất trung bình, giá trị đơn hàng cao"},
    2: {"name": "🌱 Khách Mới", "color": "#43e97b", "desc": "Ít giao dịch, chi tiêu thấp, đang khám phá"},
    3: {"name": "😴 Khách Không Hoạt Động", "color": "#f7971e", "desc": "Lịch sử mua ít, cần tái kích hoạt"},
}

SEGMENT_PERSONAS_K3 = {
    0: {"name": "💎 Khách VIP", "color": "#6c63ff", "desc": "Chi tiêu cao, mua thường xuyên"},
    1: {"name": "🔥 Khách Trung Bình", "color": "#ff6584", "desc": "Hành vi mua trung bình"},
    2: {"name": "🌱 Khách Ít Hoạt Động", "color": "#43e97b", "desc": "Chi tiêu thấp, ít giao dịch"},
}

DATA_DIR = "data/processed"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    data = {}
    files = {
        "cleaned": f"{DATA_DIR}/cleaned_uk_data.csv",
        "features": f"{DATA_DIR}/customer_features.csv",
        "scaled": f"{DATA_DIR}/customer_features_scaled.csv",
        "clusters_k3": f"{DATA_DIR}/customer_clusters_k3.csv",
        "clusters_k4": f"{DATA_DIR}/customer_clusters_k4.csv",
    }
    for key, path in files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path, index_col=0 if key in ("features", "scaled") else None)
        else:
            data[key] = None
    return data


def get_cluster_colors(k):
    return CLUSTER_COLORS_K4 if k == 4 else CLUSTER_COLORS_K3


def get_personas(k):
    return SEGMENT_PERSONAS_K4 if k == 4 else SEGMENT_PERSONAS_K3


def merge_clusters(features_df, clusters_df, k):
    if features_df is None or clusters_df is None:
        return None
    col = "Cluster"
    clusters_df = clusters_df.copy()
    clusters_df["CustomerID"] = clusters_df["CustomerID"].astype(str)
    features = features_df.copy()
    features.index = features.index.astype(str)
    merged = features.merge(clusters_df.set_index("CustomerID")[col], left_index=True, right_index=True, how="left")
    merged.rename(columns={col: f"Cluster_{k}"}, inplace=True)
    return merged


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0", family="DM Sans"),
    xaxis=dict(gridcolor="#2a2f4a", zerolinecolor="#2a2f4a"),
    yaxis=dict(gridcolor="#2a2f4a", zerolinecolor="#2a2f4a"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Segmentation")
    st.markdown("<div class='section-header'>Cài đặt phân tích</div>", unsafe_allow_html=True)

    k_choice = st.radio("Số lượng phân khúc", options=[3, 4], index=1, horizontal=True)
    st.markdown("---")

    st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "Chọn trang",
        ["📊 Tổng Quan", "🔍 Khám Phá Phân Khúc", "📈 Phân Tích RFM", "🧭 Không Gian PCA", "🏆 Xếp Hạng Khách Hàng", "👤 Tra Cứu Khách Hàng"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#8890a8;font-size:0.72rem;'>Online Retail Dataset · UK Customers<br>KMeans + PCA + Feature Engineering</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
data = load_all_data()
features_df = data["features"]
scaled_df = data["scaled"]
clusters_k = data[f"clusters_k{k_choice}"]
merged_df = merge_clusters(features_df, clusters_k, k_choice)
cleaned_df = data["cleaned"]
colors = get_cluster_colors(k_choice)
personas = get_personas(k_choice)
cluster_col = f"Cluster_{k_choice}"

# Check data availability
if features_df is None or clusters_k is None:
    st.error("⚠️ Không tìm thấy dữ liệu. Hãy chạy pipeline và đảm bảo các file CSV tồn tại trong `data/processed/`.")
    st.info(
        "Các file cần thiết:\n"
        "- `data/processed/customer_features.csv`\n"
        "- `data/processed/customer_clusters_k3.csv`\n"
        "- `data/processed/customer_clusters_k4.csv`\n"
        "- `data/processed/cleaned_uk_data.csv` (tùy chọn)"
    )
    st.stop()


# ─────────────────────────────────────────────
# ══ PAGE: TỔNG QUAN ══
# ─────────────────────────────────────────────
if page == "📊 Tổng Quan":
    st.markdown("# 📊 Tổng Quan")
    st.markdown(f"<div class='section-header'>Phân tích K = {k_choice} phân khúc khách hàng</div>", unsafe_allow_html=True)

    # ── Top KPI row
    if merged_df is not None and cluster_col in merged_df.columns:
        total_customers = len(merged_df)
        cluster_sizes = merged_df[cluster_col].value_counts().sort_index()
        largest_cluster_pct = (cluster_sizes.max() / total_customers * 100)
        avg_spend = merged_df["Sum_TotalPrice"].mean() if "Sum_TotalPrice" in merged_df.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng khách hàng", f"{total_customers:,}")
        with col2:
            st.metric("Số phân khúc", k_choice)
        with col3:
            st.metric("Chi tiêu TB (GBP)", f"£{avg_spend:,.0f}")
        with col4:
            st.metric("Phân khúc lớn nhất", f"{largest_cluster_pct:.1f}%")

        st.markdown("---")

        # ── Persona cards + distribution
        st.markdown("### Phân Khúc Khách Hàng")
        for cid, persona in personas.items():
            count = cluster_sizes.get(cid, 0)
            pct = count / total_customers * 100
            col_card, col_bar = st.columns([3, 7])
            with col_card:
                st.markdown(
                    f"""<div class='metric-card' style='border-left: 4px solid {persona["color"]}'>
                        <div class='metric-value' style='color:{persona["color"]};font-size:1.4rem'>{persona["name"]}</div>
                        <div class='metric-label'>{persona["desc"]}</div>
                        <div style='margin-top:8px;font-family:Space Mono,monospace;font-size:1.1rem'>{count:,} KH · {pct:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_bar:
                bar_fig = go.Figure(go.Bar(
                    x=[count], y=[persona["name"]], orientation="h",
                    marker_color=persona["color"], text=[f"{count:,} ({pct:.1f}%)"],
                    textposition="outside", textfont=dict(color="#e8eaf0"),
                ))
                bar_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eaf0", family="DM Sans"),
                    height=80, margin=dict(l=0, r=40, t=8, b=8),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, zerolinecolor="#2a2f4a"),
                    yaxis=dict(showticklabels=False),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("---")

        # ── Cluster size pie + feature heatmap
        col_pie, col_heat = st.columns(2)
        with col_pie:
            st.markdown("#### Tỷ lệ phân khúc")
            pie_fig = go.Figure(go.Pie(
                labels=[personas[i]["name"] for i in cluster_sizes.index if i in personas],
                values=cluster_sizes.values,
                marker_colors=[personas[i]["color"] for i in cluster_sizes.index if i in personas],
                hole=0.5,
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            pie_fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
            st.plotly_chart(pie_fig, use_container_width=True)

        with col_heat:
            st.markdown("#### Feature nổi bật theo phân khúc")
            key_features = ["Sum_TotalPrice", "Count_Invoice", "Sum_Quantity", "Mean_UnitPrice", "Count_Stock"]
            available_kf = [f for f in key_features if f in merged_df.columns]
            cluster_means = merged_df.groupby(cluster_col)[available_kf].mean()
            norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
            heat_fig = go.Figure(go.Heatmap(
                z=norm.values,
                x=[FEATURE_NAMES_VN.get(c, c) for c in norm.columns],
                y=[personas.get(int(i), {}).get("name", f"Cluster {i}") for i in norm.index],
                colorscale=[[0, "#1c2035"], [0.5, "#6c63ff"], [1, "#ff6584"]],
                text=np.round(cluster_means.values, 1),
                texttemplate="%{text:.0f}",
                textfont=dict(size=10),
            ))
            heat_fig.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(heat_fig, use_container_width=True)


# ─────────────────────────────────────────────
# ══ PAGE: KHÁM PHÁ PHÂN KHÚC ══
# ─────────────────────────────────────────────
elif page == "🔍 Khám Phá Phân Khúc":
    st.markdown("# 🔍 Khám Phá Phân Khúc")

    if merged_df is None or cluster_col not in merged_df.columns:
        st.warning("Không có dữ liệu cluster.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📡 Radar Chart", "📊 So Sánh Feature", "📋 Bảng Thống Kê"])

    # ── TAB 1: Radar Chart
    with tab1:
        radar_keys = [f for f in RADAR_FEATURES.keys() if f in merged_df.columns]
        cluster_means = merged_df.groupby(cluster_col)[radar_keys].mean()
        global_min = cluster_means.min()
        global_max = cluster_means.max()
        norm = (cluster_means - global_min) / (global_max - global_min + 1e-9)

        categories = [RADAR_FEATURES[f] for f in radar_keys]
        fig2 = go.Figure()
        for cid, row in norm.iterrows():
            vals = row.tolist() + row.tolist()[:1]
            cats = categories + categories[:1]
            color = colors.get(int(cid), "#ffffff")
            pname = personas.get(int(cid), {}).get("name", f"Cluster {cid}")
            h = color.lstrip("#")
            r2, g2, b2 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            fig2.add_trace(go.Scatterpolar(
                r=vals, theta=cats, name=pname,
                fill="toself",
                fillcolor=f"rgba({r2},{g2},{b2},0.15)",
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
            ))

        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=480,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2f4a", color="#8890a8"),
                angularaxis=dict(gridcolor="#2a2f4a", color="#e8eaf0"),
            ),
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: Feature comparison
    with tab2:
        feature_options = {FEATURE_NAMES_VN.get(f, f): f for f in features_df.columns if f in merged_df.columns}
        selected_label = st.selectbox("Chọn feature để so sánh", list(feature_options.keys()), index=3)
        selected_feat = feature_options[selected_label]

        col_box, col_violin = st.columns(2)
        with col_box:
            fig_box = go.Figure()
            for cid in sorted(merged_df[cluster_col].dropna().unique()):
                cid_int = int(cid)
                subset = merged_df[merged_df[cluster_col] == cid][selected_feat].dropna()
                p99 = subset.quantile(0.99)
                subset = subset[subset <= p99]
                fig_box.add_trace(go.Box(
                    y=subset, name=personas.get(cid_int, {}).get("name", f"C{cid}"),
                    marker_color=colors.get(cid_int, "#fff"), boxmean=True,
                ))
            fig_box.update_layout(**PLOTLY_LAYOUT, height=360, title="Box Plot")
            st.plotly_chart(fig_box, use_container_width=True)

        with col_violin:
            fig_vio = go.Figure()
            for cid in sorted(merged_df[cluster_col].dropna().unique()):
                cid_int = int(cid)
                subset = merged_df[merged_df[cluster_col] == cid][selected_feat].dropna()
                p99 = subset.quantile(0.99)
                subset = subset[subset <= p99]
                fig_vio.add_trace(go.Violin(
                    y=subset, name=personas.get(cid_int, {}).get("name", f"C{cid}"),
                    fillcolor=colors.get(cid_int, "#fff"),
                    line_color=colors.get(cid_int, "#fff"),
                    opacity=0.7, box_visible=True, meanline_visible=True,
                ))
            fig_vio.update_layout(**PLOTLY_LAYOUT, height=360, title="Violin Plot")
            st.plotly_chart(fig_vio, use_container_width=True)

        # Multi-feature bar comparison
        st.markdown("#### So sánh nhiều feature giữa các phân khúc")
        multi_feat_labels = st.multiselect(
            "Chọn các feature",
            list(feature_options.keys()),
            default=list(feature_options.keys())[:5],
        )
        if multi_feat_labels:
            multi_feats = [feature_options[l] for l in multi_feat_labels]
            avail = [f for f in multi_feats if f in merged_df.columns]
            cm = merged_df.groupby(cluster_col)[avail].mean()
            norm_cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-9)

            fig_bar = go.Figure()
            for cid in sorted(norm_cm.index):
                cid_int = int(cid)
                fig_bar.add_trace(go.Bar(
                    name=personas.get(cid_int, {}).get("name", f"C{cid}"),
                    x=[FEATURE_NAMES_VN.get(f, f) for f in avail],
                    y=norm_cm.loc[cid].values,
                    marker_color=colors.get(cid_int, "#fff"),
                    opacity=0.85,
                ))
            fig_bar.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
                                  xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 3: Stats table
    with tab3:
        st.markdown("#### Thống kê mô tả theo phân khúc")
        disp_feats = [f for f in features_df.columns if f in merged_df.columns]
        stats = merged_df.groupby(cluster_col)[disp_feats].agg(["mean", "median", "std"]).round(2)
        stats.columns = [f"{FEATURE_NAMES_VN.get(c, c)} ({s})" for c, s in stats.columns]
        stats.index = [personas.get(int(i), {}).get("name", f"Cluster {i}") for i in stats.index]
        st.dataframe(stats.T, use_container_width=True, height=500)


# ─────────────────────────────────────────────
# ══ PAGE: PHÂN TÍCH RFM ══
# ─────────────────────────────────────────────
elif page == "📈 Phân Tích RFM":
    st.markdown("# 📈 Phân Tích RFM")

    if cleaned_df is None:
        st.warning("Không tìm thấy `cleaned_uk_data.csv`. Trang này cần dữ liệu giao dịch thô.")
        st.stop()

    # Ensure datetime
    if "InvoiceDate" in cleaned_df.columns:
        cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])

    tab_rfm1, tab_rfm2, tab_rfm3 = st.tabs(["📅 Doanh Thu Theo Thời Gian", "🛍️ Sản Phẩm", "🌍 Phân Phối Khách Hàng"])

    with tab_rfm1:
        col_d, col_m = st.columns(2)
        with col_d:
            st.markdown("#### Doanh thu hàng ngày")
            daily = cleaned_df.groupby(cleaned_df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
            daily.columns = ["Date", "Revenue"]
            fig_daily = px.line(daily, x="Date", y="Revenue", color_discrete_sequence=["#6c63ff"])
            fig_daily.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Ngày", yaxis_title="Doanh thu (GBP)")
            st.plotly_chart(fig_daily, use_container_width=True)

        with col_m:
            st.markdown("#### Doanh thu hàng tháng")
            monthly = cleaned_df.groupby(cleaned_df["InvoiceDate"].dt.to_period("M").astype(str))["TotalPrice"].sum().reset_index()
            monthly.columns = ["Month", "Revenue"]
            fig_monthly = px.bar(monthly, x="Month", y="Revenue", color_discrete_sequence=["#ff6584"])
            fig_monthly.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig_monthly, use_container_width=True)

        st.markdown("#### Heatmap hoạt động theo ngày & giờ")
        if "DayOfWeek" in cleaned_df.columns and "HourOfDay" in cleaned_df.columns:
            heatmap_data = cleaned_df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        else:
            heatmap_data = cleaned_df.copy()
            heatmap_data["DayOfWeek"] = heatmap_data["InvoiceDate"].dt.dayofweek
            heatmap_data["HourOfDay"] = heatmap_data["InvoiceDate"].dt.hour
            heatmap_data = heatmap_data.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)

        day_labels = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h}:00" for h in heatmap_data.columns],
            y=[day_labels[d] if d < len(day_labels) else str(d) for d in heatmap_data.index],
            colorscale=[[0, "#1c2035"], [0.5, "#6c63ff"], [1, "#ff6584"]],
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="Giờ trong ngày", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_rfm2:
        top_n = st.slider("Số sản phẩm hiển thị", 5, 20, 10)
        col_qty, col_rev = st.columns(2)

        with col_qty:
            st.markdown(f"#### Top {top_n} sản phẩm theo số lượng")
            top_qty = cleaned_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(top_n)
            fig_qty = px.bar(
                x=top_qty.values, y=top_qty.index, orientation="h",
                color_discrete_sequence=["#6c63ff"],
                labels={"x": "Số lượng", "y": ""},
            )
            fig_qty.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_qty, use_container_width=True)

        with col_rev:
            st.markdown(f"#### Top {top_n} sản phẩm theo doanh thu")
            top_rev = cleaned_df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(top_n)
            fig_rev = px.bar(
                x=top_rev.values, y=top_rev.index, orientation="h",
                color_discrete_sequence=["#ff6584"],
                labels={"x": "Doanh thu (GBP)", "y": ""},
            )
            fig_rev.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig_rev, use_container_width=True)

    with tab_rfm3:
        st.markdown("#### Phân phối số giao dịch / khách hàng")
        txn_per_cust = cleaned_df.groupby("CustomerID")["InvoiceNo"].nunique()
        fig_hist = px.histogram(txn_per_cust[txn_per_cust <= txn_per_cust.quantile(0.99)],
                                nbins=40, color_discrete_sequence=["#43e97b"])
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Số giao dịch", yaxis_title="Số khách hàng")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Phân phối tổng chi tiêu / khách hàng")
        spend_per_cust = cleaned_df.groupby("CustomerID")["TotalPrice"].sum()
        fig_spend = px.histogram(spend_per_cust[spend_per_cust <= spend_per_cust.quantile(0.99)],
                                 nbins=40, color_discrete_sequence=["#f7971e"])
        fig_spend.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Tổng chi tiêu (GBP)", yaxis_title="Số khách hàng")
        st.plotly_chart(fig_spend, use_container_width=True)


# ─────────────────────────────────────────────
# ══ PAGE: KHÔNG GIAN PCA ══
# ─────────────────────────────────────────────
elif page == "🧭 Không Gian PCA":
    st.markdown("# 🧭 Không Gian PCA")

    if scaled_df is None or merged_df is None:
        st.warning("Không có dữ liệu scaled features.")
        st.stop()

    feat_cols = [c for c in scaled_df.columns if not c.startswith("Cluster_")]
    X = scaled_df[feat_cols].values

    @st.cache_data
    def compute_pca(X_data, n=3):
        pca = PCA(n_components=n)
        comps = pca.fit_transform(X_data)
        return comps, pca.explained_variance_ratio_

    pca_comps, evr = compute_pca(X)
    pca_df = pd.DataFrame(pca_comps, columns=["PC1", "PC2", "PC3"], index=scaled_df.index)

    # Merge cluster labels
    if cluster_col in merged_df.columns:
        pca_df[cluster_col] = merged_df[cluster_col].values

    tab_2d, tab_3d, tab_var = st.tabs(["2D Scatter", "3D Scatter", "Phương Sai PCA"])

    with tab_2d:
        st.markdown(f"PC1 ({evr[0]:.1%}) · PC2 ({evr[1]:.1%}) — Tổng: {evr[0]+evr[1]:.1%}")
        fig_2d = go.Figure()
        for cid in sorted(pca_df[cluster_col].dropna().unique()):
            cid_int = int(cid)
            sub = pca_df[pca_df[cluster_col] == cid]
            fig_2d.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"],
                mode="markers",
                name=personas.get(cid_int, {}).get("name", f"C{cid}"),
                marker=dict(color=colors.get(cid_int, "#fff"), size=5, opacity=0.7),
            ))
        fig_2d.update_layout(**PLOTLY_LAYOUT, height=480, xaxis_title="PC1", yaxis_title="PC2",
                              legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
        st.plotly_chart(fig_2d, use_container_width=True)

    with tab_3d:
        st.markdown(f"PC1 · PC2 · PC3 — Tổng: {evr[:3].sum():.1%}")
        fig_3d = go.Figure()
        for cid in sorted(pca_df[cluster_col].dropna().unique()):
            cid_int = int(cid)
            sub = pca_df[pca_df[cluster_col] == cid]
            fig_3d.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode="markers",
                name=personas.get(cid_int, {}).get("name", f"C{cid}"),
                marker=dict(color=colors.get(cid_int, "#fff"), size=3, opacity=0.7),
            ))
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0"),
            height=520,
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="PC1", gridcolor="#2a2f4a", color="#8890a8"),
                yaxis=dict(title="PC2", gridcolor="#2a2f4a", color="#8890a8"),
                zaxis=dict(title="PC3", gridcolor="#2a2f4a", color="#8890a8"),
            ),
            legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab_var:
        full_pca_comps, full_evr = compute_pca(X, n=min(16, X.shape[1]))
        cum_evr = np.cumsum(full_evr)
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=list(range(1, len(full_evr) + 1)), y=full_evr,
            name="Phương sai riêng lẻ", marker_color="#6c63ff", opacity=0.8,
        ))
        fig_var.add_trace(go.Scatter(
            x=list(range(1, len(cum_evr) + 1)), y=cum_evr,
            name="Phương sai tích lũy", line=dict(color="#ff6584", width=2), mode="lines+markers",
        ))
        fig_var.add_hline(y=0.8, line_dash="dash", line_color="#43e97b", annotation_text="80%")
        fig_var.add_hline(y=0.9, line_dash="dash", line_color="#f7971e", annotation_text="90%")
        fig_var.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title="Thành phần chính", yaxis_title="Tỷ lệ phương sai",
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_var, use_container_width=True)

        pc_table = pd.DataFrame({
            "Thành phần": [f"PC{i+1}" for i in range(len(full_evr))],
            "Phương sai (%)": [f"{v:.2%}" for v in full_evr],
            "Tích lũy (%)": [f"{v:.2%}" for v in cum_evr],
        })
        st.dataframe(pc_table, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# ══ PAGE: XẾP HẠNG KHÁCH HÀNG ══
# ─────────────────────────────────────────────
elif page == "🏆 Xếp Hạng Khách Hàng":
    st.markdown("# 🏆 Xếp Hạng Khách Hàng")
    st.markdown("<div class='section-header'>Danh sách chi tiết khách hàng theo từng phân khúc</div>", unsafe_allow_html=True)

    if merged_df is None or cluster_col not in merged_df.columns:
        st.warning("Không có dữ liệu cluster.")
        st.stop()

    tab_rank, tab_segment = st.tabs(["🥇 Xếp Hạng Chi Tiêu", "📋 Danh Sách Theo Phân Khúc"])

    # ── TAB 1: Xếp hạng chi tiêu ──────────────────────────────────
    with tab_rank:
        rank_df = merged_df.copy().reset_index()
        rank_df = rank_df.rename(columns={"index": "CustomerID"})

        # Gắn tên cluster
        rank_df["Phân Khúc"] = rank_df[cluster_col].apply(
            lambda x: personas.get(int(x), {}).get("name", f"Cluster {x}") if pd.notna(x) else "N/A"
        )
        rank_df["Màu"] = rank_df[cluster_col].apply(
            lambda x: colors.get(int(x), "#8890a8") if pd.notna(x) else "#8890a8"
        )

        # Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([3, 3, 2])
        with col_ctrl1:
            sort_feature_label = st.selectbox(
                "Xếp hạng theo",
                ["Tổng chi tiêu", "Số lần mua", "Tổng số lượng", "Giá trung bình", "Số sản phẩm khác nhau"],
            )
        with col_ctrl2:
            filter_cluster = st.multiselect(
                "Lọc phân khúc",
                options=[personas.get(i, {}).get("name", f"C{i}") for i in sorted(merged_df[cluster_col].dropna().unique().astype(int))],
                default=[personas.get(i, {}).get("name", f"C{i}") for i in sorted(merged_df[cluster_col].dropna().unique().astype(int))],
            )
        with col_ctrl3:
            sort_order = st.radio("Thứ tự", ["Cao → Thấp", "Thấp → Cao"], horizontal=True)
            top_n_rank = st.slider("Hiển thị Top", min_value=10, max_value=500, value=50, step=10)

        # Map label → feature column
        sort_map = {
            "Tổng chi tiêu": "Sum_TotalPrice",
            "Số lần mua": "Count_Invoice",
            "Tổng số lượng": "Sum_Quantity",
            "Giá trung bình": "Mean_UnitPrice",
            "Số sản phẩm khác nhau": "Count_Stock",
        }
        sort_col = sort_map[sort_feature_label]

        # Filter + sort
        ascending = sort_order == "Thấp → Cao"
        filtered = rank_df[rank_df["Phân Khúc"].isin(filter_cluster)]
        if sort_col in filtered.columns:
            filtered = filtered.sort_values(sort_col, ascending=ascending)

        filtered = filtered.head(top_n_rank).reset_index(drop=True)
        filtered.index = filtered.index + 1  # bắt đầu từ 1

        # ── Bar chart xếp hạng
        st.markdown(f"#### Top {top_n_rank} khách hàng theo {sort_feature_label}")
        if sort_col in filtered.columns and len(filtered) > 0:
            fig_rank = go.Figure()
            for cid_int in sorted(merged_df[cluster_col].dropna().unique().astype(int)):
                sub = filtered[filtered[cluster_col] == cid_int]
                if len(sub) == 0:
                    continue
                pname = personas.get(cid_int, {}).get("name", f"C{cid_int}")
                color = colors.get(cid_int, "#fff")
                fig_rank.add_trace(go.Bar(
                    x=sub["CustomerID"].astype(str),
                    y=sub[sort_col],
                    name=pname,
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate=(
                        "<b>KH: %{x}</b><br>"
                        + f"{sort_feature_label}: " + "%{y:,.1f}<br>"
                        + f"Phân khúc: {pname}"
                        + "<extra></extra>"
                    ),
                ))
            prefix = "£" if "Price" in sort_col or "price" in sort_col else ""
            fig_rank.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaf0", family="DM Sans"),
                margin=dict(l=40, r=20, t=40, b=40),
                height=420,
                barmode="stack",
                xaxis=dict(showticklabels=False, title="Customer ID", gridcolor="#2a2f4a", zerolinecolor="#2a2f4a"),
                yaxis=dict(title=f"{prefix}{sort_feature_label}", gridcolor="#2a2f4a", zerolinecolor="#2a2f4a"),
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        # ── Bảng chi tiết
        st.markdown("#### Bảng chi tiết")
        display_cols_rank = ["CustomerID", "Phân Khúc"]
        feat_show = [f for f in ["Sum_TotalPrice", "Count_Invoice", "Sum_Quantity", "Mean_UnitPrice", "Count_Stock"] if f in filtered.columns]
        display_cols_rank += feat_show

        table_df = filtered[display_cols_rank].copy()
        rename_map = {"CustomerID": "Customer ID", "Phân Khúc": "Phân Khúc"}
        rename_map.update({f: FEATURE_NAMES_VN.get(f, f) for f in feat_show})
        table_df = table_df.rename(columns=rename_map)

        # Format số
        for col_name in ["Tổng chi tiêu", "Giá trung bình"]:
            if col_name in table_df.columns:
                table_df[col_name] = table_df[col_name].apply(lambda x: f"£{x:,.2f}")
        for col_name in ["Số lần mua", "Tổng số lượng", "Số sản phẩm khác nhau"]:
            if col_name in table_df.columns:
                table_df[col_name] = table_df[col_name].apply(lambda x: f"{x:,.0f}")

        st.dataframe(table_df, use_container_width=True, height=420)

        # Download button
        csv_data = filtered[display_cols_rank].to_csv(index=True).encode("utf-8")
        st.download_button(
            label="⬇️ Tải xuống CSV",
            data=csv_data,
            file_name=f"customer_ranking_{sort_feature_label.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # ── TAB 2: Danh sách theo phân khúc ──────────────────────────
    with tab_segment:
        st.markdown("#### Khách hàng thuộc từng phân khúc")

        seg_df = merged_df.copy().reset_index()
        seg_df = seg_df.rename(columns={"index": "CustomerID"})
        seg_df["Phân Khúc"] = seg_df[cluster_col].apply(
            lambda x: personas.get(int(x), {}).get("name", f"Cluster {x}") if pd.notna(x) else "N/A"
        )

        # Chọn phân khúc muốn xem
        selected_seg = st.selectbox(
            "Chọn phân khúc",
            options=[personas.get(i, {}).get("name", f"C{i}") for i in sorted(merged_df[cluster_col].dropna().unique().astype(int))],
        )

        seg_filtered = seg_df[seg_df["Phân Khúc"] == selected_seg].copy()

        # Tìm cluster id để lấy màu
        seg_cid = next(
            (i for i in sorted(merged_df[cluster_col].dropna().unique().astype(int))
             if personas.get(i, {}).get("name") == selected_seg), 0
        )
        seg_color = colors.get(seg_cid, "#6c63ff")

        # KPI row cho phân khúc đó
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Số khách hàng", f"{len(seg_filtered):,}")
        with c2:
            avg_sp = seg_filtered["Sum_TotalPrice"].mean() if "Sum_TotalPrice" in seg_filtered.columns else 0
            st.metric("Chi tiêu TB", f"£{avg_sp:,.0f}")
        with c3:
            avg_inv = seg_filtered["Count_Invoice"].mean() if "Count_Invoice" in seg_filtered.columns else 0
            st.metric("Số mua TB", f"{avg_inv:,.1f} đơn")
        with c4:
            avg_qty = seg_filtered["Sum_Quantity"].mean() if "Sum_Quantity" in seg_filtered.columns else 0
            st.metric("Số lượng TB", f"{avg_qty:,.0f} sp")

        st.markdown("---")

        # Sort option
        col_s1, col_s2 = st.columns([4, 2])
        with col_s1:
            seg_sort_label = st.selectbox(
                "Sắp xếp theo",
                ["Tổng chi tiêu", "Số lần mua", "Tổng số lượng", "Giá trung bình", "Số sản phẩm khác nhau"],
                key="seg_sort",
            )
        with col_s2:
            seg_sort_order = st.radio("Thứ tự", ["Cao → Thấp", "Thấp → Cao"], horizontal=True, key="seg_order")

        seg_sort_col = sort_map[seg_sort_label]
        seg_asc = seg_sort_order == "Thấp → Cao"

        if seg_sort_col in seg_filtered.columns:
            seg_filtered = seg_filtered.sort_values(seg_sort_col, ascending=seg_asc).reset_index(drop=True)
            seg_filtered.index = seg_filtered.index + 1

        # Bảng
        feat_seg = [f for f in ["Sum_TotalPrice", "Count_Invoice", "Sum_Quantity", "Mean_UnitPrice", "Count_Stock"] if f in seg_filtered.columns]
        seg_display = seg_filtered[["CustomerID"] + feat_seg].copy()
        seg_rename = {"CustomerID": "Customer ID"}
        seg_rename.update({f: FEATURE_NAMES_VN.get(f, f) for f in feat_seg})
        seg_display = seg_display.rename(columns=seg_rename)

        for col_name in ["Tổng chi tiêu", "Giá trung bình"]:
            if col_name in seg_display.columns:
                seg_display[col_name] = seg_display[col_name].apply(lambda x: f"£{x:,.2f}")
        for col_name in ["Số lần mua", "Tổng số lượng", "Số sản phẩm khác nhau"]:
            if col_name in seg_display.columns:
                seg_display[col_name] = seg_display[col_name].apply(lambda x: f"{x:,.0f}")

        st.dataframe(seg_display, use_container_width=True, height=500)

        # Download
        csv_seg = seg_filtered[["CustomerID"] + feat_seg].to_csv(index=True).encode("utf-8")
        st.download_button(
            label=f"⬇️ Tải xuống danh sách {selected_seg}",
            data=csv_seg,
            file_name=f"customers_{selected_seg.replace(' ', '_').replace('💎','').replace('🔥','').replace('🌱','').replace('😴','').strip()}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────
# ══ PAGE: TRA CỨU KHÁCH HÀNG ══
# ─────────────────────────────────────────────
elif page == "👤 Tra Cứu Khách Hàng":
    st.markdown("# 👤 Tra Cứu Khách Hàng")

    if merged_df is None or cluster_col not in merged_df.columns:
        st.warning("Không có dữ liệu cluster.")
        st.stop()

    all_customers = sorted(merged_df.index.astype(str).tolist())

    col_search, col_rand = st.columns([5, 1])
    with col_search:
        selected_customer = st.selectbox("Nhập hoặc chọn Customer ID", all_customers)
    with col_rand:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🎲 Ngẫu nhiên"):
            selected_customer = np.random.choice(all_customers)

    if selected_customer:
        try:
            row = merged_df.loc[selected_customer]
        except KeyError:
            st.error(f"Không tìm thấy khách hàng: {selected_customer}")
            st.stop()

        cid_int = int(row[cluster_col]) if pd.notna(row[cluster_col]) else -1
        persona = personas.get(cid_int, {"name": "Unknown", "color": "#8890a8", "desc": ""})

        st.markdown(
            f"""<div class='metric-card' style='border-left:4px solid {persona["color"]};margin-bottom:24px'>
                <div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#8890a8;margin-bottom:4px'>CUSTOMER ID</div>
                <div style='font-family:Space Mono,monospace;font-size:1.8rem;font-weight:700'>{selected_customer}</div>
                <div style='margin-top:8px'>
                  <span class='cluster-badge' style='background:{persona["color"]}22;color:{persona["color"]};border:1px solid {persona["color"]}44'>
                    {persona["name"]}
                  </span>
                  <span style='color:#8890a8;font-size:0.82rem;margin-left:10px'>{persona["desc"]}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Feature values
        feat_cols_display = [f for f in features_df.columns if f in merged_df.columns]
        cust_vals = row[feat_cols_display]
        cluster_means_ref = merged_df.groupby(cluster_col)[feat_cols_display].mean()
        overall_means = merged_df[feat_cols_display].mean()

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Chỉ số chính")
            key_metrics = [
                ("Sum_TotalPrice", "Tổng chi tiêu", "£", ""),
                ("Count_Invoice", "Số lần mua", "", " đơn"),
                ("Sum_Quantity", "Tổng số lượng", "", " sp"),
                ("Count_Stock", "Sản phẩm đa dạng", "", " loại"),
                ("Mean_UnitPrice", "Giá TB", "£", ""),
            ]
            for feat, label, prefix, suffix in key_metrics:
                if feat in cust_vals.index:
                    val = cust_vals[feat]
                    mean_val = overall_means[feat]
                    delta_pct = (val - mean_val) / (mean_val + 1e-9) * 100
                    st.metric(label, f"{prefix}{val:,.1f}{suffix}", delta=f"{delta_pct:+.1f}% vs TB")

        with col_right:
            st.markdown("#### So sánh với trung bình phân khúc")
            if cid_int in cluster_means_ref.index:
                cluster_mean_row = cluster_means_ref.loc[cid_int]
                radar_keys = [f for f in RADAR_FEATURES.keys() if f in feat_cols_display]

                # Normalize globally
                global_max_r = merged_df[radar_keys].max()
                global_min_r = merged_df[radar_keys].min()
                cust_norm = (cust_vals[radar_keys] - global_min_r) / (global_max_r - global_min_r + 1e-9)
                cl_norm = (cluster_mean_row[radar_keys] - global_min_r) / (global_max_r - global_min_r + 1e-9)

                cats = [RADAR_FEATURES[f] for f in radar_keys]
                fig_cust_radar = go.Figure()
                for norm_vals, trace_name, color, opacity in [
                    (cust_norm.tolist(), f"KH {selected_customer}", persona["color"], 0.3),
                    (cl_norm.tolist(), "TB Phân khúc", "#8890a8", 0.1),
                ]:
                    h = color.lstrip("#")
                    rc, gc, bc = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    vals = norm_vals + norm_vals[:1]
                    c_list = cats + cats[:1]
                    fig_cust_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=c_list, name=trace_name,
                        fill="toself", fillcolor=f"rgba({rc},{gc},{bc},{opacity})",
                        line=dict(color=color, width=2),
                    ))

                fig_cust_radar.update_layout(
                    **PLOTLY_LAYOUT, height=340,
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2f4a", color="#8890a8"),
                        angularaxis=dict(gridcolor="#2a2f4a", color="#e8eaf0"),
                    ),
                    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                )
                st.plotly_chart(fig_cust_radar, use_container_width=True)

        # All features bar
        st.markdown("#### Tất cả features (so với TB phân khúc)")
        if cid_int in cluster_means_ref.index:
            compare_df = pd.DataFrame({
                "Feature": [FEATURE_NAMES_VN.get(f, f) for f in feat_cols_display],
                "Khách hàng": cust_vals[feat_cols_display].values,
                "TB Phân khúc": cluster_means_ref.loc[cid_int, feat_cols_display].values,
            })
            # Normalize for display
            max_vals = compare_df[["Khách hàng", "TB Phân khúc"]].max(axis=1)
            compare_norm = compare_df.copy()
            compare_norm["Khách hàng"] = compare_df["Khách hàng"] / (max_vals + 1e-9)
            compare_norm["TB Phân khúc"] = compare_df["TB Phân khúc"] / (max_vals + 1e-9)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name=f"KH {selected_customer}", x=compare_norm["Feature"],
                                     y=compare_norm["Khách hàng"], marker_color=persona["color"], opacity=0.85))
            fig_cmp.add_trace(go.Bar(name="TB Phân khúc", x=compare_norm["Feature"],
                                     y=compare_norm["TB Phân khúc"], marker_color="#8890a8", opacity=0.6))
            fig_cmp.update_layout(**PLOTLY_LAYOUT, height=340, barmode="group",
                                  xaxis_tickangle=-35, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Transaction history
        if cleaned_df is not None:
            st.markdown("#### Lịch sử giao dịch")
            cust_txns = cleaned_df[cleaned_df["CustomerID"].astype(str) == str(selected_customer)]

            if len(cust_txns) > 0:
                st.markdown(f"**{len(cust_txns):,} giao dịch** · Tổng: £{cust_txns['TotalPrice'].sum():,.2f}")

                display_cols = [
                    c for c in ["InvoiceDate", "InvoiceNo", "Description", "Quantity", "UnitPrice", "TotalPrice"]
                    if c in cust_txns.columns
                ]

                st.dataframe(
                    cust_txns[display_cols].sort_values("InvoiceDate", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Không tìm thấy lịch sử giao dịch cho khách hàng này.")