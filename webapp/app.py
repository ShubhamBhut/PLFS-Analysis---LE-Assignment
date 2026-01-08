import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Urban Women Labor Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (MODERN UI & VISIBILITY FIX) ---
st.markdown("""
    <style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* GLASSMORPHISM CARDS for Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05); /* Translucent White */
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Force Label Colors for High Contrast */
    div[data-testid="stMetricLabel"] p {
        color: #b0b0b0 !important; /* Light Grey for Title */
        font-size: 0.9rem !important;
    }
    
    div[data-testid="stMetricValue"] div {
        color: #ffffff !important; /* Pure White for Numbers */
        font-weight: 700 !important;
    }

    /* Custom Headers */
    h1, h2, h3 {
        color: #f0f2f6;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111111;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed_urban_women_2324.csv')
    except:
        return pd.DataFrame() # Fallback

    if 'total_wage' in df.columns: df['total_wage'] = df['total_wage'].clip(lower=0)
    if 'spousal_wage' in df.columns: df['spousal_wage'] = df['spousal_wage'].clip(lower=0)
    
    caste_map = {1: 'Scheduled Tribe (ST)', 2: 'Scheduled Caste (SC)', 3: 'OBC', 9: 'General/Others'}
    relg_map = {1: 'Hinduism', 2: 'Islam', 3: 'Christianity', 4: 'Sikhism', 9: 'Others'}
    
    def simplify_edu(code):
        try:
            c = int(code)
            if c <= 4: return '1. Low/Illiterate'
            if c <= 8: return '2. Middle/Secondary'
            if c <= 12: return '3. Graduate/Tech'
            return '4. Post-Grad+'
        except: return 'Unknown'

    if 'sg' in df.columns: df['Caste'] = df['sg'].map(caste_map).fillna('General/Others')
    if 'relg' in df.columns: df['Religion'] = df['relg'].map(relg_map).fillna('Others')
    if 'gedu_lvl' in df.columns: df['Education_Group'] = df['gedu_lvl'].apply(simplify_edu)
    
    return df

df_master = load_data()

if not df_master.empty:
    # --- 2. SIDEBAR (CLEAN, NO ICON) ---
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("---")
    
    # Filters
    all_castes = df_master['Caste'].unique() if 'Caste' in df_master.columns else []
    selected_caste = st.sidebar.multiselect("Social Group", all_castes, default=all_castes)
    
    all_edu = df_master['Education_Group'].unique() if 'Education_Group' in df_master.columns else []
    selected_edu = st.sidebar.multiselect("Education Level", all_edu, default=all_edu)
    
    age_range = st.sidebar.slider("Age Demographics", 15, 60, (15, 60))

    df_filtered = df_master[
        (df_master['Caste'].isin(selected_caste)) &
        (df_master['Education_Group'].isin(selected_edu)) &
        (df_master['age'].between(age_range[0], age_range[1]))
    ]

    # --- 3. MAIN DASHBOARD ---
    st.title("🇮🇳 Urban Women Workforce Dynamics")
    st.markdown("### Strategic Analytics Dashboard")
    st.markdown("---")

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participation Rate", f"{df_filtered['is_employed'].mean()*100:.1f}%", help="Percentage of women currently working")
    
    working_only = df_filtered[df_filtered['total_wage'] > 0]
    median_wage = working_only['total_wage'].median() if not working_only.empty else 0
    c2.metric("Median Monthly Wage", f"₹{median_wage:,.0f}", help="Median earnings for employed women")
    
    c3.metric("Avg Household Size", f"{df_filtered['hh_size'].mean():.1f}")
    
    gap = df_filtered['spousal_wage'].median() - median_wage
    c4.metric("Spousal Wage Gap", f"₹{gap:,.0f}", delta=-gap, delta_color="inverse", help="Difference between Husband and Wife Median Wage")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Macro Trends", 
        "🏠 Household", 
        "💰 Wage Analysis", 
        "🧬 Deep Tech View", 
        "🤖 ML Simulator"
    ])

    # --- TAB 1: MACRO ---
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Lifecycle Participation Curve")
            lifecycle = df_filtered.groupby('age')['is_employed'].mean().reset_index()
            fig_life = px.line(lifecycle, x='age', y='is_employed', markers=True, line_shape='spline')
            fig_life.update_traces(line_color='#00d2ff', line_width=4)
            fig_life.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            fig_life.add_vrect(x0=23, x1=30, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Drop-off Zone")
            st.plotly_chart(fig_life, use_container_width=True)
        with c2:
            st.subheader("Social Group Share")
            sg_part = df_filtered.groupby('Caste')['is_employed'].mean().reset_index()
            fig_bar = px.bar(sg_part, x='Caste', y='is_employed', color='is_employed', color_continuous_scale='Viridis')
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: HOUSEHOLD ---
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Income Effect")
            df_filtered['Spouse_Bin'] = pd.qcut(df_filtered['spousal_wage'].rank(method='first'), q=10, labels=range(1, 11))
            inc_data = df_filtered.groupby('Spouse_Bin')['is_employed'].mean().reset_index()
            fig_inc = px.bar(inc_data, x='Spouse_Bin', y='is_employed', color='is_employed', color_continuous_scale='Magma')
            fig_inc.update_layout(plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_inc, use_container_width=True)
        with c2:
            st.subheader("Child Penalty")
            hh_data = df_filtered[df_filtered['hh_size']<=8].groupby('hh_size')['is_employed'].mean().reset_index()
            fig_hh = px.line(hh_data, x='hh_size', y='is_employed', markers=True)
            fig_hh.update_traces(line_color='#ff4b4b', line_width=3)
            fig_hh.update_layout(plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_hh, use_container_width=True)

    # --- TAB 3: WAGES ---
    with tab3:
        st.subheader("The Mincer Earnings Function")
        if not working_only.empty:
            wage_data = working_only.groupby(['age', 'Education_Group'])['total_wage'].median().reset_index()
            fig_w = px.line(wage_data, x='age', y='total_wage', color='Education_Group', line_shape='spline')
            fig_w.update_layout(plot_bgcolor="rgba(0,0,0,0)", font_color="white", hovermode="x unified")
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.warning("Insufficient data.")

    # --- TAB 4: DEEP TECH VIEW (IMPRESSIVE PLOTS) ---
    with tab4:
        st.subheader("🧬 Multi-Dimensional Capital Flow (Sankey Diagram)")
        st.markdown("Tracking the flow of population from **Social Group** → **Education** → **Employment Status**.")
        
        # Prepare Data for Sankey
        # 1. Source: Caste -> Target: Education
        sankey_df1 = df_filtered.groupby(['Caste', 'Education_Group']).size().reset_index(name='value')
        sankey_df1.columns = ['source', 'target', 'value']
        
        # 2. Source: Education -> Target: Employment
        df_filtered['Status'] = df_filtered['is_employed'].map({1: 'Working', 0: 'Not Working'})
        sankey_df2 = df_filtered.groupby(['Education_Group', 'Status']).size().reset_index(name='value')
        sankey_df2.columns = ['source', 'target', 'value']
        
        links = pd.concat([sankey_df1, sankey_df2], axis=0)
        
        # Get unique labels
        unique_nodes = list(pd.concat([links['source'], links['target']]).unique())
        node_map = {name: i for i, name in enumerate(unique_nodes)}
        
        links['source_id'] = links['source'].map(node_map)
        links['target_id'] = links['target'].map(node_map)
        
        # Plot Sankey
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
              label = unique_nodes,
              color = "blue"
            ),
            link = dict(
              source = links['source_id'],
              target = links['target_id'],
              value = links['value'],
              color = 'rgba(200, 200, 200, 0.3)'
            ))])
        fig_sankey.update_layout(height=600, font_size=12, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🧊 3D Labor Hyperplane")
        st.markdown("Interactive analysis of **Age**, **Spousal Wage**, and **Individual Wage** in 3D space.")
        
        # 3D Scatter Plot (Sampled for performance)
        sample_3d = working_only.sample(min(1000, len(working_only)))
        fig_3d = px.scatter_3d(sample_3d, x='age', y='spousal_wage', z='total_wage',
                               color='Education_Group', size='hh_size', opacity=0.7,
                               title="The Wage Hypercube")
        fig_3d.update_layout(height=600, margin=dict(l=0, r=0, b=0, t=0), 
                             scene=dict(bgcolor='rgba(0,0,0,0)', xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
                                        yaxis=dict(backgroundcolor='rgba(0,0,0,0)'), zaxis=dict(backgroundcolor='rgba(0,0,0,0)')))
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAB 5: ML SIMULATOR ---
    with tab5:
        st.subheader("🤖 AI Predictive Engine")
        
        # ML Logic
        le = LabelEncoder()
        df_master['Education_Encoded'] = le.fit_transform(df_master['Education_Group'])
        features = ['age', 'hh_size', 'spousal_wage', 'Education_Encoded']
        
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(df_master[features], df_master['is_employed'])
        
        reg = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        workers_reg = df_master[df_master['total_wage'] > 0]
        reg.fit(workers_reg[features], workers_reg['total_wage'])
        
        c1, c2, c3 = st.columns(3)
        with c1:
            age_in = st.number_input("Age", 18, 60, 24)
            edu_in = st.selectbox("Education", sorted(df_master['Education_Group'].unique()))
        with c2:
            hh_in = st.slider("Household Size", 1, 10, 4)
            spouse_in = st.number_input("Spouse Income (₹)", 0, 500000, 20000)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Simulation", type="primary"):
                # Predict
                edu_code = le.transform([edu_in])[0]
                input_vec = pd.DataFrame([[age_in, hh_in, spouse_in, edu_code]], columns=features)
                
                prob = clf.predict_proba(input_vec)[0][1]
                wage = reg.predict(input_vec)[0]
                
                st.success(f"Employment Probability: **{prob*100:.1f}%**")
                st.info(f"Predicted Wage: **₹{wage:,.0f}**")