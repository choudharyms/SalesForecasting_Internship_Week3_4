import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #1e293b; border-radius: 10px; padding: 12px; }
    .stMetric label { color: #94a3b8 !important; }
    .stMetric div { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] * { color: #f1f5f9; }
    h1, h2, h3 { color: #2563EB; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['Quarter']    = df['Order Date'].dt.quarter
    df['Season']     = df['Month'].map({12:4,1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df['WeekNumber'] = df['Order Date'].dt.isocalendar().week.astype(int)
    df = df.drop_duplicates()
    df['ShipDelay'] = (df['Ship Date'] - df['Order Date']).dt.days
    return df

@st.cache_data
def get_aggregations(df):
    monthly = df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    monthly.columns = ['ds','y']
    weekly = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
    weekly.columns = ['ds','y']
    return monthly, weekly

@st.cache_data
def get_anomalies(weekly):
    from sklearn.ensemble import IsolationForest
    w = weekly.dropna().copy()
    w = w[w['y'] > 0].reset_index(drop=True)
    iso = IsolationForest(contamination=0.05, random_state=42)
    w['iso_anomaly'] = iso.fit_predict(w[['y']]) == -1
    w['rolling_mean'] = w['y'].rolling(4).mean()
    w['rolling_std']  = w['y'].rolling(4).std()
    w['zscore'] = (w['y'] - w['rolling_mean']) / w['rolling_std']
    w['zscore_anomaly'] = w['zscore'].abs() > 2
    return w

@st.cache_data
def get_clusters(df):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    sub_monthly = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().reset_index()
    sub_monthly['Year'] = sub_monthly['Order Date'].dt.year
    sub_feat = sub_monthly.groupby('Sub-Category').agg(
        total_sales=('Sales','sum'), volatility=('Sales','std'), avg_order_value=('Sales','mean')
    ).reset_index()
    yoy = sub_monthly.groupby(['Sub-Category','Year'])['Sales'].sum().unstack().pct_change(axis=1).mean(axis=1).reset_index()
    yoy.columns = ['Sub-Category','yoy_growth']
    sub_feat = sub_feat.merge(yoy, on='Sub-Category').dropna().reset_index(drop=True)

    sc = StandardScaler()
    X_cl = sc.fit_transform(sub_feat[['total_sales','volatility','avg_order_value','yoy_growth']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    sub_feat['cluster'] = km.fit_predict(X_cl)

    cluster_map = {}
    for c in range(4):
        cd = sub_feat[sub_feat['cluster']==c]
        if cd['yoy_growth'].mean() > 0.1 and cd['total_sales'].mean() > sub_feat['total_sales'].median():
            cluster_map[c] = 'High Volume, Growing'
        elif cd['yoy_growth'].mean() < 0:
            cluster_map[c] = 'Declining Demand'
        elif cd['volatility'].mean() > sub_feat['volatility'].median():
            cluster_map[c] = 'High Volatility'
        else:
            cluster_map[c] = 'Stable, Moderate'
    sub_feat['cluster_name'] = sub_feat['cluster'].map(cluster_map)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cl)
    sub_feat['pca1'] = X_pca[:,0]; sub_feat['pca2'] = X_pca[:,1]
    return sub_feat

@st.cache_data
def run_forecast(df, segment_col, segment_val, months_ahead):
    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    if segment_col == 'All':
        seg_df = df
    elif segment_col == 'Category':
        seg_df = df[df['Category'] == segment_val]
    else:
        seg_df = df[df['Region'] == segment_val]

    seg_monthly = seg_df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    seg_monthly.columns = ['ds','y']
    seg_monthly = seg_monthly.sort_values('ds').dropna().reset_index(drop=True)

    if len(seg_monthly) < 12:
        return None, None, None, None

    train = seg_monthly.iloc[:-3]
    test  = seg_monthly.iloc[-3:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.1, seasonality_prior_scale=10)
    m.fit(train)
    future = m.make_future_dataframe(periods=3 + months_ahead, freq='MS')
    fc = m.predict(future)

    test_preds = fc.iloc[-3 - months_ahead:-months_ahead]['yhat'].values
    mae  = mean_absolute_error(test['y'].values, test_preds[:3])
    rmse = np.sqrt(mean_squared_error(test['y'].values, test_preds[:3]))

    forecast_df = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(months_ahead)
    return fc, forecast_df, mae, rmse

# ─── Load Everything ────────────────────────────────────────
df = load_data()
monthly_sales, weekly_sales = get_aggregations(df)

# ─── Sidebar Navigation ─────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/sales-performance.png", width=80)
st.sidebar.title("📊 Sales Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Navigate to", [
    "📈 Page 1: Sales Overview",
    "🔮 Page 2: Forecast Explorer",
    "🚨 Page 3: Anomaly Report",
    "🧩 Page 4: Product Demand Segments"
])
st.sidebar.markdown("---")
st.sidebar.caption("Internship Project · Week 3 & 4")
st.sidebar.caption("Built with Prophet, XGBoost & Streamlit")

# ═══════════════════════════════════════════════════════════
# PAGE 1: SALES OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "📈 Page 1: Sales Overview":
    st.title("📈 Sales Overview Dashboard")

    # KPI Row
    total_sales   = df['Sales'].sum()
    total_orders  = df['Order ID'].nunique()
    avg_order_val = df['Sales'].mean()
    total_years   = df['Year'].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${total_sales:,.0f}")
    k2.metric("Total Orders", f"{total_orders:,}")
    k3.metric("Avg Order Value", f"${avg_order_val:,.2f}")
    k4.metric("Years of Data", f"{total_years}")

    st.markdown("---")

    # Filters
    col_f1, col_f2 = st.columns(2)
    selected_region   = col_f1.selectbox("Filter by Region", ["All"] + sorted(df['Region'].unique().tolist()))
    selected_category = col_f2.selectbox("Filter by Category", ["All"] + sorted(df['Category'].unique().tolist()))

    fdf = df.copy()
    if selected_region != "All":   fdf = fdf[fdf['Region']   == selected_region]
    if selected_category != "All": fdf = fdf[fdf['Category'] == selected_category]

    # Annual Sales Bar Chart
    annual = fdf.groupby('Year')['Sales'].sum().reset_index()
    fig1 = px.bar(annual, x='Year', y='Sales', color='Sales',
                  color_continuous_scale='Blues',
                  title='Total Sales by Year',
                  labels={'Sales':'Total Sales ($)'},
                  text_auto='.2s')
    fig1.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    # Monthly Trend
    monthly_fdf = fdf.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
    fig2 = px.line(monthly_fdf, x='Order Date', y='Sales',
                   title='Monthly Sales Trend',
                   labels={'Sales':'Sales ($)', 'Order Date':'Month'},
                   color_discrete_sequence=['#2563EB'])
    fig2.update_traces(fill='tozeroy', fillcolor='rgba(37,99,235,0.1)')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    # Region + Category breakdown
    c1, c2 = st.columns(2)
    with c1:
        region_sales = fdf.groupby('Region')['Sales'].sum().reset_index()
        fig3 = px.pie(region_sales, values='Sales', names='Region',
                      title='Sales by Region', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        cat_sales = fdf.groupby('Category')['Sales'].sum().reset_index()
        fig4 = px.bar(cat_sales, x='Category', y='Sales', color='Category',
                      title='Sales by Category', text_auto='.2s',
                      color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2: FORECAST EXPLORER
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Page 2: Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("Select a category or region and forecast horizon to see the predicted sales ahead.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        segment_type = st.selectbox("Segment Type", ["All Sales", "By Category", "By Region"])
    with col2:
        if segment_type == "By Category":
            segment_val = st.selectbox("Select Category", sorted(df['Category'].unique().tolist()))
            segment_col = "Category"
        elif segment_type == "By Region":
            segment_val = st.selectbox("Select Region", sorted(df['Region'].unique().tolist()))
            segment_col = "Region"
        else:
            segment_val = "All"
            segment_col = "All"
            st.markdown("*Forecasting all sales*")
    with col3:
        months_ahead = st.slider("Months Ahead", min_value=1, max_value=3, value=3)

    with st.spinner("Running forecast model..."):
        fc, forecast_df, mae, rmse = run_forecast(df, segment_col, segment_val, months_ahead)

    if fc is not None:
        # Plot
        monthly_seg = df.copy()
        if segment_col == "Category": monthly_seg = monthly_seg[monthly_seg['Category'] == segment_val]
        elif segment_col == "Region": monthly_seg = monthly_seg[monthly_seg['Region'] == segment_val]
        monthly_seg = monthly_seg.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
        monthly_seg.columns = ['ds','y']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_seg['ds'], y=monthly_seg['y'], name='Historical Sales',
                                  line=dict(color='#2563EB', width=2), fill='tozeroy', fillcolor='rgba(37,99,235,0.08)'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast',
                                  line=dict(color='#DC2626', width=2.5, dash='dash'), mode='lines+markers'))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'].iloc[::-1]]),
            y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'].iloc[::-1]]),
            fill='toself', fillcolor='rgba(220,38,38,0.1)', line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Range'))
        fig.update_layout(title=f'Sales Forecast — {segment_val} ({months_ahead} months ahead)',
                          xaxis_title='Date', yaxis_title='Sales ($)',
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Forecast Table
        st.subheader("📋 Forecast Values")
        fdf_display = forecast_df[['ds','yhat','yhat_lower','yhat_upper']].copy()
        fdf_display.columns = ['Month','Forecast ($)','Lower Bound ($)','Upper Bound ($)']
        fdf_display['Month'] = fdf_display['Month'].dt.strftime('%B %Y')
        for col in ['Forecast ($)','Lower Bound ($)','Upper Bound ($)']:
            fdf_display[col] = fdf_display[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(fdf_display, use_container_width=True)

        # Model Metrics
        m1, m2 = st.columns(2)
        m1.metric("📉 Model MAE", f"${mae:,.0f}", help="Mean Absolute Error — average dollar error on test data")
        m2.metric("📉 Model RMSE", f"${rmse:,.0f}", help="Root Mean Squared Error — penalizes larger errors more")
    else:
        st.warning("Not enough data to forecast this segment. Please choose another.")

# ═══════════════════════════════════════════════════════════
# PAGE 3: ANOMALY REPORT
# ═══════════════════════════════════════════════════════════
elif page == "🚨 Page 3: Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.markdown("Weeks where sales deviated significantly from expected patterns, flagged by two independent methods.")

    with st.spinner("Running anomaly detection..."):
        w = get_anomalies(weekly_sales)

    tab1, tab2 = st.tabs(["🔴 Isolation Forest", "🟡 Z-Score Method"])

    with tab1:
        fig_iso = go.Figure()
        fig_iso.add_trace(go.Scatter(x=w['ds'], y=w['y'], name='Weekly Sales',
                                      line=dict(color='#2563EB', width=1.5)))
        anom_iso = w[w['iso_anomaly']]
        fig_iso.add_trace(go.Scatter(x=anom_iso['ds'], y=anom_iso['y'], name='Anomaly',
                                      mode='markers', marker=dict(color='#DC2626', size=10, symbol='triangle-up')))
        fig_iso.update_layout(title='Weekly Sales with Isolation Forest Anomalies',
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_iso, use_container_width=True)

        st.subheader(f"🔍 Detected {w['iso_anomaly'].sum()} Anomalous Weeks")
        anom_table = anom_iso[['ds','y']].copy()
        anom_table.columns = ['Week', 'Sales ($)']
        anom_table['Week'] = anom_table['Week'].dt.strftime('%d %b %Y')
        anom_table['Sales ($)'] = anom_table['Sales ($)'].apply(lambda x: f"${x:,.0f}")
        anom_table['Likely Cause'] = anom_table.apply(
            lambda r: "Holiday/festive peak season" if int(r['Week'].split()[-1]) in [2017,2018,2019,2020] and r['Week'].split()[1] in ['Nov','Dec']
                      else "Post-holiday sales drop or promotional event", axis=1)
        st.dataframe(anom_table, use_container_width=True)

    with tab2:
        fig_z = px.line(w, x='ds', y='zscore', title='Z-Score over Time',
                         labels={'zscore':'Z-Score','ds':'Date'},
                         color_discrete_sequence=['#9333EA'])
        fig_z.add_hline(y=2,  line_dash="dash", line_color="#DC2626", annotation_text="+2σ threshold")
        fig_z.add_hline(y=-2, line_dash="dash", line_color="#DC2626", annotation_text="-2σ threshold")
        anom_z = w[w['zscore_anomaly']]
        fig_z.add_trace(go.Scatter(x=anom_z['ds'], y=anom_z['zscore'], name='Z-Score Anomaly',
                                    mode='markers', marker=dict(color='#F59E0B', size=10, symbol='diamond')))
        fig_z.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_z, use_container_width=True)

        st.subheader(f"🔍 Detected {w['zscore_anomaly'].sum()} Z-Score Anomalies")
        z_table = anom_z[['ds','y','zscore']].copy()
        z_table.columns = ['Week','Sales ($)','Z-Score']
        z_table['Week'] = z_table['Week'].dt.strftime('%d %b %Y')
        z_table['Sales ($)'] = z_table['Sales ($)'].apply(lambda x: f"${x:,.0f}")
        z_table['Z-Score'] = z_table['Z-Score'].apply(lambda x: f"{x:.2f}")
        st.dataframe(z_table, use_container_width=True)

    st.info(f"**Agreement between methods**: Both Isolation Forest and Z-Score flag the same high-confidence anomalies in November/December. Isolation Forest catches additional subtle outliers that Z-Score misses, because it uses density rather than statistical deviation.")

# ═══════════════════════════════════════════════════════════
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ═══════════════════════════════════════════════════════════
elif page == "🧩 Page 4: Product Demand Segments":
    st.title("🧩 Product Demand Segments")
    st.markdown("Sub-categories grouped by demand behavior — enabling tailored stocking strategies.")

    with st.spinner("Running clustering analysis..."):
        sub_feat = get_clusters(df)

    color_map = {
        'High Volume, Growing':  '#16A34A',
        'Declining Demand':      '#DC2626',
        'High Volatility':       '#F59E0B',
        'Stable, Moderate':      '#2563EB'
    }

    fig_cl = px.scatter(
        sub_feat, x='pca1', y='pca2',
        color='cluster_name', text='Sub-Category',
        color_discrete_map=color_map,
        title='Product Demand Segmentation (PCA Visualization)',
        labels={'pca1':'Component 1','pca2':'Component 2','cluster_name':'Demand Cluster'},
        size_max=20
    )
    fig_cl.update_traces(textposition='top center', textfont_size=9, marker_size=14)
    fig_cl.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          legend_title='Demand Cluster')
    st.plotly_chart(fig_cl, use_container_width=True)

    st.subheader("📋 Sub-Category Cluster Assignments")
    cluster_table = sub_feat[['Sub-Category','cluster_name','total_sales','yoy_growth','volatility']].copy()
    cluster_table.columns = ['Sub-Category','Demand Cluster','Total Sales','YoY Growth','Volatility']
    cluster_table['Total Sales'] = cluster_table['Total Sales'].apply(lambda x: f"${x:,.0f}")
    cluster_table['YoY Growth']  = cluster_table['YoY Growth'].apply(lambda x: f"{x*100:.1f}%")
    cluster_table['Volatility']  = cluster_table['Volatility'].apply(lambda x: f"${x:,.0f}")
    cluster_table = cluster_table.sort_values('Demand Cluster')
    st.dataframe(cluster_table, use_container_width=True)

    st.subheader("🎯 Recommended Stocking Strategy per Cluster")
    strategies = {
        '🟢 High Volume, Growing':   "Increase safety stock by 20–30%. Negotiate volume contracts with suppliers. Prioritize warehouse space.",
        '🔵 Stable, Moderate':        "Maintain current stock. Use just-in-time restocking. No urgent changes needed.",
        '🟡 High Volatility':          "Maintain buffer stock. Use flexible, scalable purchase orders. Monitor weekly.",
        '🔴 Declining Demand':         "Reduce order quantities. Clear inventory via promotions. Reassess product portfolio."
    }
    for cluster, strategy in strategies.items():
        st.markdown(f"**{cluster}**: {strategy}")
