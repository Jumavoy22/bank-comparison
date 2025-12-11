import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Helper Functions & Constants ---

def gini(array):
    """Calculates the Gini coefficient for a numpy array."""
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    # Convert to float to avoid casting errors
    array = array.astype(float)
    
    if array.size == 0:
        return np.nan
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

METRICS_CONFIG = {
    'total_tx': {'name': 'Total Transactions', 'better_is': 'higher'},
    'total_vol': {'name': 'Total Volume', 'better_is': 'higher'},
    'avg_check': {'name': 'Average Check', 'better_is': 'higher'},
    'median_check': {'name': 'Median Check', 'better_is': 'higher'},
    'p95': {'name': '95th Percentile of Sum', 'better_is': 'higher'},
    'high_value_share': {'name': 'High-Value Txn. Share (> Market p95)', 'better_is': 'higher'},
    'p2p_share': {'name': 'P2P Transfer Share', 'better_is': 'lower'},
    'p2p_avg': {'name': 'Average P2P Check', 'better_is': 'higher'},
    'top_cat_share': {'name': 'Top Category Share', 'better_is': 'lower'},
    'top_issue_method_share': {'name': 'Top Issue Method Share', 'better_is': 'lower'},
    'micro_share': {'name': 'Micro-Transaction Share (<10% of Median)', 'better_is': 'lower'},
    'gini': {'name': 'Gini Coefficient', 'better_is': 'lower'},
}

@st.cache_data
def load_and_clean_data(uploaded_file):
    """Loads data from a CSV and applies default cleaning."""
    try:
        df = pd.read_csv(uploaded_file)
        # --- Mandatory columns ---
        df['trans_date'] = pd.to_datetime(df['trans_date'])
        df['total_sum'] = pd.to_numeric(df['total_sum'], errors='coerce').fillna(0)
        
        # --- Optional columns ---
        if 'p2p_flag' in df.columns:
            if df['p2p_flag'].dtype == 'object':
                df['p2p_flag'] = df['p2p_flag'].str.lower().isin(['true', '1', 't'])
            else:
                df['p2p_flag'] = df['p2p_flag'].astype(bool)
        else:
            df['p2p_flag'] = False
        
      
        
        # Critical fields
        crit_cols = [col for col in ["transaction_code", "trans_date", "total_sum", "bank_name"] if col in df.columns]
        df.dropna(subset=crit_cols, inplace=True)
        
        # Text fields
        if 'bank_name' in df.columns:
            df["bank_name"] = df["bank_name"].str.strip().str.upper()
        if 'emitent_region' in df.columns:
            df["emitent_region"] = df["emitent_region"].str.strip().str.title()
        if 'gender' in df.columns:
            df["gender"] = df["gender"].str.strip().str.lower()

        # Flags for p2p
        if 'p2p_flag' in df.columns:
            df["p2p_flag"] = df["p2p_flag"].fillna(0).astype(bool)

        # Outlier filter
        if 'total_sum' in df.columns and not df.empty:
            df = df[df["total_sum"] > 0]
            low, high = df["total_sum"].quantile([0.01, 0.99])
            df = df[(df["total_sum"] >= low) & (df["total_sum"] <= high)]

        return df

    except KeyError as e:
        st.error(f"Error: The file is missing a required column: {e}. "
                 f"Please ensure the file contains at least 'trans_date', 'total_sum', and 'bank_name'.")
        return None
    except Exception as e:
        st.error(f"Error reading or cleaning file: {e}")
        return None

@st.cache_data
def calculate_metrics(df):
    """Calculates key metrics for each bank, handling missing columns."""
    if df.empty:
        return pd.DataFrame(columns=['bank_name'] + list(METRICS_CONFIG.keys())).set_index('bank_name')

    # --- Basic aggregations ---
    agg_dict = {
        'total_vol': ('total_sum', 'sum'),
        'avg_check': ('total_sum', 'mean'),
        'median_check': ('total_sum', 'median'),
        'p95': ('total_sum', lambda x: x.quantile(0.95)),
    }
    if 'transaction_code' in df.columns:
        agg_dict['total_tx'] = ('transaction_code', 'count')

    bank_metrics = df.groupby('bank_name').agg(**agg_dict).reset_index()
    
    market_p95 = df['total_sum'].quantile(0.95)

    other_metrics = []
    for bank_name, group in df.groupby('bank_name'):
        metric_row = {'bank_name': bank_name}
        median_check_val = bank_metrics.loc[bank_metrics['bank_name'] == bank_name, 'median_check'].values[0]

        # --- Safely calculate each metric ---
        metric_row['high_value_share'] = (group['total_sum'] > market_p95).mean()
        metric_row['micro_share'] = (group['total_sum'] < 0.1 * median_check_val).mean() if median_check_val > 0 else 0
        
        if 'p2p_flag' in group.columns:
            p2p_group = group[group['p2p_flag']]
            metric_row['p2p_share'] = len(p2p_group) / len(group) if len(group) > 0 else 0
            metric_row['p2p_avg'] = p2p_group['total_sum'].mean() if not p2p_group.empty else 0
        
        if 'trans_category' in group.columns and not group['trans_category'].empty:
            metric_row['top_cat_share'] = group['trans_category'].value_counts(normalize=True).max()
        
        if 'issue_method' in group.columns and not group['issue_method'].empty:
            metric_row['top_issue_method_share'] = group['issue_method'].value_counts(normalize=True).max()
            
        if 'emitent_region' in group.columns and not group['emitent_region'].empty:
            metric_row['top_region_share'] = group['emitent_region'].value_counts(normalize=True).max()
        
        metric_row['gini'] = gini(group['total_sum'])
        
        other_metrics.append(metric_row)

    if other_metrics:
        other_metrics_df = pd.DataFrame(other_metrics)
        bank_metrics = pd.merge(bank_metrics, other_metrics_df, on='bank_name', how='left')

    for metric in METRICS_CONFIG:
        if metric not in bank_metrics.columns:
            bank_metrics[metric] = 0

    return bank_metrics.set_index('bank_name').fillna(0)


@st.cache_data
def get_ranked_data(metrics_df):
    """Adds ranks and percentiles to the metrics dataframe."""
    ranked_df = metrics_df.copy()
    num_banks = len(ranked_df)
    if num_banks == 0:
        return ranked_df

    for metric, config in METRICS_CONFIG.items():
        if metric in ranked_df.columns:
            ascending = (config['better_is'] == 'lower')
            ranked_df[f'{metric}_rank'] = ranked_df[metric].rank(method='min', ascending=ascending).astype(int)
            ranked_df[f'{metric}_percentile'] = (1 - (ranked_df[f'{metric}_rank'] - 1) / num_banks) * 100
    return ranked_df

# --- Main Application ---

st.title("📊 Competitive Bank Analysis")

# --- Sidebar ---
st.sidebar.header("Filters and Settings")
uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type="csv")

df_full = None
if uploaded_file is not None:
    df_full = load_and_clean_data(uploaded_file)
else:
    try:
        df_full = load_and_clean_data('vertolet.csv')
        st.sidebar.success("Loaded and cleaned demo file `vertolet.csv`.")
    except FileNotFoundError:
        st.info("Please upload a CSV file to begin analysis.")
        st.stop()

if df_full is None:
    st.stop()

bank_names = sorted(df_full['bank_name'].unique())

# --- Data-based Filters ---
min_date, max_date = df_full['trans_date'].min(), df_full['trans_date'].max()

selected_date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)
if len(selected_date_range) != 2:
    st.stop()

selected_categories = []
if 'trans_category' in df_full.columns:
    all_categories = sorted(df_full['trans_category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Transaction Category", options=all_categories, default=all_categories
    )

# --- Filtering Data ---
start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
df_filtered = df_full[
    (df_full['trans_date'] >= start_date) & (df_full['trans_date'] <= end_date)
]

if selected_categories and 'trans_category' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['trans_category'].isin(selected_categories)]


if df_filtered.empty:
    st.warning("No data matching the selected filters. Please adjust the filter settings.")
    st.stop()

# --- Calculations ---
metrics_df = calculate_metrics(df_filtered)
ranked_df = get_ranked_data(metrics_df)

# --- Main View ---
st.header("Competitor Comparison")
my_bank = st.selectbox("Select your bank for comparison:", options=bank_names)

if my_bank and my_bank in ranked_df.index:
    st.subheader(f"Key Metrics for: {my_bank}")

    my_bank_data = ranked_df.loc[[my_bank]]
    summary_list = []
    # Display only calculated metrics
    for metric, config in METRICS_CONFIG.items():
        if f'{metric}_rank' in my_bank_data.columns and not my_bank_data[metric].iloc[0] == 0:
            summary_list.append({
                'Metric': config['name'],
                'Value': my_bank_data[metric].iloc[0],
                'Rank': f"{my_bank_data[f'{metric}_rank'].iloc[0]} of {len(bank_names)}",
                'Percentile': f"{my_bank_data[f'{metric}_percentile'].iloc[0]:.1f}%",
                'Better when': 'Higher' if config['better_is'] == 'higher' else 'Lower'
            })
    summary_df = pd.DataFrame(summary_list)
    
    format_dict = {'Value': '{:,.2f}'}
    if not summary_df.empty:
        st.dataframe(summary_df.style.format(format_dict), use_container_width=True)

    st.subheader("Visual Comparison by Metric")
    
    cols = st.columns(3)
    col_idx = 0
    
    # Display only calculated metrics
    sorted_metrics = sorted(METRICS_CONFIG.items(), key=lambda item: item[0])

    for metric, config in sorted_metrics:
        if metric in ranked_df.columns and not ranked_df[metric].sum() == 0:
            plot_df = ranked_df.sort_values(by=metric, ascending=False)
            plot_colors = ['#d62728' if bank == my_bank else '#1f77b4' for bank in plot_df.index]

            fig = px.bar(
                plot_df, x=plot_df.index, y=metric,
                title=config['name'], labels={'x': 'Bank', metric: 'Value'},
                text_auto='.2s'
            )
            fig.update_traces(marker_color=plot_colors, textposition='outside')
            fig.update_layout(
                title_font_size=16, xaxis_title=None, yaxis_title=None,
                showlegend=False, height=300, margin=dict(t=40, b=0, l=0, r=0)
            )
            cols[col_idx % 3].plotly_chart(fig, use_container_width=True)
            col_idx += 1

st.sidebar.markdown("---")
st.sidebar.info("Dashboard for transaction data analysis.")
