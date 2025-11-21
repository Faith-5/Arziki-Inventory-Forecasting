import streamlit as st
import pandas as pd
import json
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import your pipeline
from src.pipeline import ArzikiPipeline
from src.utils.logger import setup_logger

# Page configuration
st.set_page_config(
    page_title="Arziki ML - Inventory Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'outputs' not in st.session_state:
    st.session_state.outputs = None
if 'df_forecast' not in st.session_state:
    st.session_state.df_forecast = None
if 'df_insights' not in st.session_state:
    st.session_state.df_insights = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

def reset_session():
    """Reset all session state variables"""
    st.session_state.pipeline_run = False
    st.session_state.outputs = None
    st.session_state.df_forecast = None
    st.session_state.df_insights = None
    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
        shutil.rmtree(st.session_state.temp_dir)
    st.session_state.temp_dir = None

def run_pipeline(uploaded_file, forecast_horizon):
    """Execute the Arziki pipeline"""
    try:
        # Create temporary directory for this session
        temp_dir = tempfile.mkdtemp()
        st.session_state.temp_dir = temp_dir
        
        # Save uploaded file
        input_path = Path(temp_dir) / "input.csv"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Setup logger
        logger = setup_logger("arziki_streamlit")
        
        # Initialize and run pipeline
        output_dir = Path(temp_dir) / "output"
        pipeline = ArzikiPipeline(
            output_dir=str(output_dir),
            logger=logger,
            forecast_horizon=forecast_horizon
        )
        
        with st.spinner("🔄 Running ML pipeline... This may take a few minutes."):
            outputs = pipeline.run(str(input_path))
        
        # Load outputs into session state
        st.session_state.outputs = outputs
        st.session_state.df_forecast = pd.read_json(outputs["forecast"])
        st.session_state.df_insights = pd.read_json(outputs["insights"])
        st.session_state.pipeline_run = True
        
        return True, "Pipeline executed successfully!"
    
    except Exception as e:
        return False, f"Pipeline execution failed: {str(e)}"

def display_forecast_tab():
    """Display forecast results"""
    df = st.session_state.df_forecast.copy()
    
    st.subheader("📈 Demand Forecasts")
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if 'Product Name' in df.columns:
            products = ['All Products'] + sorted(df['Product Name'].unique().tolist())
            selected_product = st.selectbox("Filter by Product", products)
    
    with col2:
        if 'Category' in df.columns:
            categories = ['All Categories'] + sorted(df['Category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
    
    with col3:
        st.metric("Total Products", df['Product Name'].nunique() if 'Product Name' in df.columns else 0)
    
    # Apply filters
    if selected_product != 'All Products':
        df = df[df['Product Name'] == selected_product]
    if selected_category != 'All Categories':
        df = df[df['Category'] == selected_category]
    
    # Display summary statistics
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Forecast Rows", len(df))
        with col2:
            if 'Predicted Units' in df.columns:
                st.metric("Avg Predicted Units", f"{df['Predicted Units'].mean():.2f}")
        with col3:
            if 'Predicted Units Daily' in df.columns:
                st.metric("Avg Daily Demand", f"{df['Predicted Units Daily'].mean():.2f}")
        with col4:
            if 'Date' in df.columns:
                st.metric("Forecast Period", f"{len(df['Date'].unique())} days")
    
    # Display dataframe
    st.dataframe(
        df.style.format({
            'Predicted Units': '{:.2f}',
            'Predicted Units Daily': '{:.2f}',
            'Lower Bound': '{:.2f}',
            'Upper Bound': '{:.2f}'
        } if all(col in df.columns for col in ['Predicted Units', 'Predicted Units Daily']) else {}),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def display_inventory_tab():
    """Display inventory metrics"""
    df_insights = st.session_state.df_insights.copy()
    
    st.subheader("📦 Inventory Metrics")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if 'Product Name' in df_insights.columns:
            products = ['All Products'] + sorted(df_insights['Product Name'].unique().tolist())
            selected_product = st.selectbox("Filter by Product", products, key="inv_product")
    
    with col2:
        risk_filter = st.selectbox("Filter by Risk", ['All', 'Stockout Warning', 'Overstock Risk', 'Healthy'])
    
    # Apply filters
    if selected_product != 'All Products':
        df_insights = df_insights[df_insights['Product Name'] == selected_product]
    
    if risk_filter != 'All':
        if risk_filter == 'Stockout Warning':
            df_insights = df_insights[df_insights['Stockout Warning'] == True]
        elif risk_filter == 'Overstock Risk':
            df_insights = df_insights[df_insights['Overstock Risk'] == True]
        elif risk_filter == 'Healthy':
            df_insights = df_insights[
                (df_insights['Stockout Warning'] == False) & 
                (df_insights['Overstock Risk'] == False)
            ]
    
    # Display metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stockout_count = df_insights['Stockout Warning'].sum() if 'Stockout Warning' in df_insights.columns else 0
        st.metric("⚠️ Stockout Warnings", int(stockout_count))
    with col2:
        overstock_count = df_insights['Overstock Risk'].sum() if 'Overstock Risk' in df_insights.columns else 0
        st.metric("📦 Overstock Risks", int(overstock_count))
    with col3:
        healthy_count = len(df_insights) - stockout_count - overstock_count
        st.metric("✅ Healthy Inventory", int(healthy_count))
    with col4:
        avg_days = df_insights['Days to Stockout'].mean() if 'Days to Stockout' in df_insights.columns else 0
        st.metric("Avg Days to Stockout", f"{avg_days:.1f}")
    
    # Display dataframe with color coding
    st.dataframe(
        df_insights.style.format({
            'Avg Daily Demand': '{:.2f}',
            'Safety Stock': '{:.2f}',
            'ROP': '{:.2f}',
            'EOQ': '{:.2f}',
            'Days to Stockout': '{:.1f}'
        } if 'Avg Daily Demand' in df_insights.columns else {}),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = df_insights.to_csv(index=False)
    st.download_button(
        label="📥 Download Inventory Metrics CSV",
        data=csv,
        file_name=f"inventory_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def display_insights_tab():
    """Display actionable insights"""
    df_insights = st.session_state.df_insights.copy()
    
    st.subheader("💡 Actionable Insights")
    
    # Priority insights
    if 'Stockout Warning' in df_insights.columns:
        critical = df_insights[df_insights['Stockout Warning'] == True]
        if not critical.empty:
            st.error(f"🚨 **{len(critical)} products** need immediate attention (stockout risk)")
            with st.expander("View Critical Products"):
                st.dataframe(critical[['Product Name', 'Days to Stockout', 'Insight']], use_container_width=True)
    
    if 'Overstock Risk' in df_insights.columns:
        overstock = df_insights[df_insights['Overstock Risk'] == True]
        if not overstock.empty:
            st.warning(f"📦 **{len(overstock)} products** have overstock risk")
            with st.expander("View Overstock Products"):
                st.dataframe(overstock[['Product Name', 'Insight']], use_container_width=True)
    
    # All insights
    st.markdown("---")
    st.markdown("### All Product Insights")
    
    for _, row in df_insights.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                product_name = row.get('Product Name', 'Unknown Product')
                insight_text = row.get('Insight', 'No insight available')
                
                # Determine icon based on risk
                if row.get('Stockout Warning', False):
                    icon = "⚠️"
                    color = "red"
                elif row.get('Overstock Risk', False):
                    icon = "📦"
                    color = "orange"
                else:
                    icon = "✅"
                    color = "green"
                
                st.markdown(f"**{icon} {product_name}**")
                st.markdown(f"<span style='color:{color}'>{insight_text}</span>", unsafe_allow_html=True)
            
            with col2:
                if 'Days to Stockout' in row:
                    st.metric("Days to Stockout", f"{row['Days to Stockout']:.1f}")
            
            st.markdown("---")
    
    # Download insights JSON
    if st.session_state.outputs:
        with open(st.session_state.outputs["insights"], 'r') as f:
            insights_json = f.read()
        
        st.download_button(
            label="📥 Download Insights JSON",
            data=insights_json,
            file_name=f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def display_charts_tab():
    """Display generated charts"""
    st.subheader("📊 Visualizations")
    
    charts_dir = Path(st.session_state.outputs["charts_dir"])
    
    if not charts_dir.exists():
        st.warning("No charts directory found")
        return
    
    # Get all chart files
    chart_files = list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.html"))
    
    if not chart_files:
        st.info("No charts generated yet. Charts will appear here after pipeline execution.")
        return
    
    # Organize charts by type
    product_charts = [f for f in chart_files if 'product' in f.stem.lower()]
    category_charts = [f for f in chart_files if 'category' in f.stem.lower()]
    other_charts = [f for f in chart_files if f not in product_charts and f not in category_charts]
    
    # Display category charts first
    if category_charts:
        st.markdown("### 📈 Category-Level Trends")
        for chart_path in category_charts:
            st.markdown(f"**{chart_path.stem.replace('_', ' ').title()}**")
            if chart_path.suffix == '.png':
                img = Image.open(chart_path)
                st.image(img, use_container_width=True)
            else:
                with open(chart_path, 'r') as f:
                    st.components.v1.html(f.read(), height=600)
            
            # Download button
            with open(chart_path, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {chart_path.name}",
                    data=f.read(),
                    file_name=chart_path.name,
                    mime="image/png" if chart_path.suffix == '.png' else "text/html"
                )
            st.markdown("---")
    
    # Display product-specific charts
    if product_charts:
        st.markdown("### 🏷️ Product-Level Charts")
        
        # Add product selector if many charts
        if len(product_charts) > 5:
            chart_names = [c.stem for c in product_charts]
            selected_chart = st.selectbox("Select Product Chart", chart_names)
            selected_chart_path = [c for c in product_charts if c.stem == selected_chart][0]
            product_charts = [selected_chart_path]
        
        for chart_path in product_charts:
            st.markdown(f"**{chart_path.stem.replace('_', ' ').title()}**")
            if chart_path.suffix == '.png':
                img = Image.open(chart_path)
                st.image(img, use_container_width=True)
            else:
                with open(chart_path, 'r') as f:
                    st.components.v1.html(f.read(), height=600)
            
            # Download button
            with open(chart_path, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {chart_path.name}",
                    data=f.read(),
                    file_name=chart_path.name,
                    mime="image/png" if chart_path.suffix == '.png' else "text/html",
                    key=f"download_{chart_path.stem}"
                )
            st.markdown("---")
    
    # Display other charts
    if other_charts:
        st.markdown("### 📉 Additional Visualizations")
        for chart_path in other_charts:
            st.markdown(f"**{chart_path.stem.replace('_', ' ').title()}**")
            if chart_path.suffix == '.png':
                img = Image.open(chart_path)
                st.image(img, use_container_width=True)
            else:
                with open(chart_path, 'r') as f:
                    st.components.v1.html(f.read(), height=600)
            
            # Download button
            with open(chart_path, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {chart_path.name}",
                    data=f.read(),
                    file_name=chart_path.name,
                    mime="image/png" if chart_path.suffix == '.png' else "text/html",
                    key=f"download_other_{chart_path.stem}"
                )
            st.markdown("---")

def main():
    # Header
    st.markdown('<div class="main-header">🛒 Arziki ML - Inventory Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict demand, prevent stockouts, optimize inventory</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        
        st.markdown("### 📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload your inventory CSV file",
            type=['csv'],
            help="Upload a CSV file containing your supermarket inventory/sales data"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        
        forecast_horizon = st.slider(
            "Forecast Horizon (weeks)",
            min_value=1,
            max_value=12,
            value=4,
            help="Number of weeks to forecast into the future"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload** your inventory CSV file
        2. **Configure** forecast settings
        3. **Run** the pipeline
        4. **Explore** forecasts, metrics & insights
        5. **Download** results
        
        **Expected CSV Columns:**
        - Product Name
        - Category
        - Date
        - Units Sold
        - Stock on Hand
        - Unit Cost
        - Selling Price
        - Supplier Lead Time
        """)
        
        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_session()
            st.rerun()
    
    # Main content
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file to begin")
        
        # Show sample data structure
        st.markdown("### 📊 Sample Data Structure")
        sample_data = {
            'Product Name': ['Product A', 'Product B', 'Product C'],
            'Category': ['Category 1', 'Category 2', 'Category 1'],
            'Date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'Units Sold': [100, 150, 80],
            'Stock on Hand': [500, 300, 200],
            'Unit Cost': [10.0, 15.0, 12.0],
            'Selling Price': [15.0, 22.0, 18.0],
            'Supplier Lead Time': [7, 14, 10]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
    else:
        # Show file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        # Run pipeline button
        if not st.session_state.pipeline_run:
            if st.button("🚀 Run ML Pipeline", type="primary", use_container_width=True):
                success, message = run_pipeline(uploaded_file, forecast_horizon)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # Display results if pipeline has run
        if st.session_state.pipeline_run:
            st.success("✅ Pipeline completed successfully!")
            
            # Tabs for different views
            tabs = st.tabs(["📈 Forecasts", "📦 Inventory Metrics", "💡 Insights", "📊 Charts"])
            
            with tabs[0]:
                display_forecast_tab()
            
            with tabs[1]:
                display_inventory_tab()
            
            with tabs[2]:
                display_insights_tab()
            
            with tabs[3]:
                display_charts_tab()

if __name__ == "__main__":
    main()