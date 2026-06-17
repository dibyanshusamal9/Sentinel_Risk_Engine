import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import base64

st.set_page_config(page_title="Fraud Detection Engine", layout="wide")

def inject_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

h1, h2, h3, h4, h5, h6, p, li, label, div[data-testid="stMarkdownContainer"] * {
    font-family: 'Nunito', sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    letter-spacing: -0.01em !important;
}
h1 {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    margin-bottom: 0.5rem !important;
}
.stMarkdown p, .stMarkdown li {
    font-size: 1.1rem;
    max-width: 75ch; /* Optimal reading length */
}

section[data-testid="stSidebar"] * {
    font-size: 14px !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2 {
    font-size: 1.75rem !important;
    line-height: 1.2 !important;
    margin-bottom: 1rem !important;
    white-space: nowrap !important;
    overflow: hidden !important; 
    text-overflow: ellipsis !important;
}

section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[kind="secondary"] {
    min-height: 2rem !important;
    height: auto !important;
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
    margin-top: 0 !important;
    margin-bottom: 0.25rem !important;
}

section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}

.block-container {
    animation: fadeIn 0.6s cubic-bezier(0.25, 1, 0.5, 1);
}

[data-testid="stAlert"], [data-testid="stAlert"] > div, [data-testid="stDataFrame"] > div > div, .stDataFrame > div {
    border-radius: 12px !important;
}
[data-testid="stDataFrame"] > div {
    box-shadow: 0 20px 40px rgba(0,0,0,0.04) !important;
}

div[data-baseweb="select"] input {
    pointer-events: none !important;
    caret-color: transparent !important;
}

[data-testid="stMetric"] {
    animation: slideUpFadeIn 0.5s cubic-bezier(0.25, 1, 0.5, 1);
    animation-fill-mode: both;
    background-color: var(--secondary-background-color);
    padding: 1.25rem;
    border-radius: 12px !important;
    border-left: 4px solid var(--primary-color) !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.04);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.08);
}
[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { animation-delay: 0.1s; border-left-color: #64748b !important; }
[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { animation-delay: 0.2s; border-left-color: #10b981 !important; }
[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { animation-delay: 0.3s; border-left-color: #f59e0b !important; }
[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { animation-delay: 0.4s; border-left-color: #e11d48 !important; }

.stButton button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-radius: 12px !important;
    border: 2px solid var(--primary-color) !important;
    min-height: 48px;
    transition: transform 0.3s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.3s cubic-bezier(0.25, 1, 0.5, 1) !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.04) !important;
}
.stButton button:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.12) !important;
}
.stButton button:active {
    transform: translateY(0);
    box-shadow: 0 5px 10px rgba(0,0,0,0.05) !important;
}
.stButton button:focus-visible {
    outline: 3px solid var(--primary-color) !important;
    outline-offset: 2px !important;
}

hr {
    border-color: rgba(59, 130, 246, 0.15) !important;
}

/* Keyframes */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUpFadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* Style material icons */
html body span.material-symbols-rounded,
html body span.material-icons,
html body div[data-baseweb="radio"] span.material-symbols-rounded,
html body div[data-baseweb="popover"] span.material-symbols-rounded,
html body [data-testid="stSidebarCollapseButton"] span,
html body [data-testid="stToolbar"] span,
html body button[kind="header"] span {
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    line-height: 1 !important;
    white-space: nowrap !important;
}

</style>
""", unsafe_allow_html=True)

inject_custom_css()

# Load assets
@st.cache_resource
def load_ml_artifacts():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"We couldn't load the machine learning models. Please verify that `model.pkl` and `scaler.pkl` are in your project folder. \n\nError details: {str(e)}")
        st.stop()

@st.cache_data
def load_rules():
    try:
        with open("rules.yaml", "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"We couldn't load the business rules. Please check that `rules.yaml` is in your project folder and is properly formatted. \n\nError details: {str(e)}")
        st.stop()

@st.cache_data
def load_test_stream():
    try:
        return pd.read_csv("external_test_dataset.zip", compression='zip')
    except Exception as e:
        st.error(f"We couldn't load the transaction data. Please ensure `external_test_dataset.zip` is in your project folder. \n\nError details: {str(e)}")
        st.stop()

# Initialize system
model, scaler = load_ml_artifacts()
rules = load_rules()
df = load_test_stream()

# Navigation setup
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

st.sidebar.markdown(
    """
    <div style="margin-bottom: 1.25rem;">
        <h2 style="font-size: 1.45rem !important; font-weight: 700 !important; line-height: 1.2 !important; margin: 0 !important; white-space: normal !important; overflow: visible !important; letter-spacing: -0.01em !important;">Sentinel Risk<br>Engine</h2>
        <span style="font-size: 0.8rem !important; opacity: 0.6 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; font-weight: 600 !important;">Threat Intelligence</span>
    </div>
    <hr style="margin-top: 0; margin-bottom: 1.25rem; border-color: rgba(59, 130, 246, 0.15) !important;">
    <p style="font-size: 0.85rem !important; font-weight: 700 !important; opacity: 0.5 !important; margin-bottom: 0.5rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important;">Navigation</p>
    """,
    unsafe_allow_html=True
)
st.sidebar.button("Home", use_container_width=True, type="primary" if st.session_state.page == "Home" else "secondary", on_click=set_page, args=("Home",))
st.sidebar.button("About Project", use_container_width=True, type="primary" if st.session_state.page == "About Project" else "secondary", on_click=set_page, args=("About Project",))
st.sidebar.button("Fraud Scoring Engine", use_container_width=True, type="primary" if st.session_state.page == "Fraud Scoring Engine" else "secondary", on_click=set_page, args=("Fraud Scoring Engine",))
st.sidebar.button("Business Rules", use_container_width=True, type="primary" if st.session_state.page == "Business Rules" else "secondary", on_click=set_page, args=("Business Rules",))
st.sidebar.button("Dataset Overview", use_container_width=True, type="primary" if st.session_state.page == "Dataset Overview" else "secondary", on_click=set_page, args=("Dataset Overview",))

page = st.session_state.page

if page == "Home":
    # Background image injection
    try:
        with open("bg.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
                background-color: transparent !important;
            }}
            [data-testid="stMain"] {{
                overflow: hidden !important;
            }}
            [data-testid="stMainBlockContainer"] {{
                padding-top: 2rem !important;
            }}
            .bg-overlay {{
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                z-index: -998;
                background-color: rgba(240, 242, 246, 0.88);
            }}
            @media (prefers-color-scheme: dark) {{
                .bg-overlay {{
                    background-color: rgba(15, 23, 42, 0.88);
                }}
            }}
            @media (prefers-color-scheme: light) {{
                .bg-overlay {{
                    background-color: rgba(255, 255, 255, 0.88);
                }}
            }}
            </style>
            <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -999; background-image: url('data:image/png;base64,{encoded_string}'); background-size: cover; background-position: center; background-repeat: no-repeat;"></div>
            <div class="bg-overlay"></div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass # Fail gracefully if image doesn't exist
        

    st.markdown("""
    <style>
    @media (prefers-color-scheme: dark) {
        .hero-text-container, .hero-text-container h1, .hero-text-container h3, .hero-text-container p {
            color: #ffffff !important;
        }
        [data-testid="stHeader"], [data-testid="stHeader"] button, [data-testid="stHeader"] span {
            color: #ffffff !important;
        }
    }
    @media (prefers-color-scheme: light) {
        .hero-text-container, .hero-text-container h1, .hero-text-container h3, .hero-text-container p {
            color: #0f172a !important;
        }
        [data-testid="stHeader"], [data-testid="stHeader"] button, [data-testid="stHeader"] span {
            color: #0f172a !important;
        }
    }
    </style>
    <div class="hero-text-container">
        <h1 style="font-size: 2.5rem; font-weight: 700; padding-bottom: 0.5rem;">Sentinel Risk Engine</h1>
        <h3 style="font-size: 1.25rem; font-weight: 600; padding-bottom: 1rem;">Real-Time Transaction Monitoring & Threat Defense</h3>
        <p style="font-size: 1rem; line-height: 1.6;">Welcome to the live dashboard. Sentinel continuously monitors incoming transaction streams, applying a dual-layer defense mechanism to instantly isolate financial threats before they settle.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    

    st.markdown('<div class="hero-text-container"><h3 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">Quick Start</h3></div>', unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        st.markdown("""
        <div class="hero-text-container">
            <h3 style="font-size: 1.1rem; font-weight: 600;">Run the Scoring Engine</h3>
            <p style="font-size: 0.95rem; margin-bottom: 1rem;">Inject streaming batches of transaction data and watch the hybrid system evaluate threats in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Launch Engine", type="primary", on_click=set_page, args=("Fraud Scoring Engine",), use_container_width=True)
        
    with col_nav2:
        st.markdown("""
        <div class="hero-text-container">
            <h3 style="font-size: 1.1rem; font-weight: 600;">Read the Documentation</h3>
            <p style="font-size: 0.95rem; margin-bottom: 1rem;">Deep dive into the Sentinel architecture, the F2 mathematical optimization, and the dataset engineering.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("View Documentation", type="secondary", on_click=set_page, args=("About Project",), use_container_width=True)
    
elif page == "About Project":
    st.title("About the Project")
    st.markdown("""
    **A hybrid threat-detection architecture bridging the gap between predictive machine learning and real-world business logic.**
    
    [![View on GitHub](https://img.shields.io/badge/View_Full_Project-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/dibyanshusamal9/Sentinel_Risk_Engine)
    """)
    st.divider()

    # Architecture diagram
    st.graphviz_chart('''
        digraph Architecture {
            rankdir=LR;
            node [shape=rect, style="rounded,filled", fontname="Nunito", fillcolor="#ffffff", color="#e2e8f0", penwidth=2, margin=0.2];
            edge [color="#94a3b8", penwidth=2, arrowsize=0.8];
            
            Stream [label="Raw Transaction\\nStream"];
            FE [label="Feature Engineering\\n(Temporal Extraction)"];
            ML [label="XGBoost AI Layer\\n(46.4% Threshold)"];
            Rules [label="YAML Rules Engine\\n(Business Overrides)"];
            Output [label="Decision Engine\\n(Block / Flag / Approve)", fillcolor="#fee2e2", color="#ef4444"];
            
            Stream -> FE -> ML -> Rules -> Output;
        }
    ''')
    st.divider()

    # Phase 1
    st.subheader("Phase 1: The Data & Feature Engineering")
    st.markdown("""
    The foundation of this engine is based on a highly anonymized European credit card dataset. In financial fraud detection, privacy is paramount, which drastically limits the data available to the model.
    
    * **PCA Transformation:** To protect Personally Identifiable Information (PII), the original features (like location, merchant ID, and device type) were mathematically compressed into 28 anonymous Principal Components (`V1` through `V28`).
    * **Temporal Engineering:** The dataset provided time as a continuous counter of seconds. Because machine learning models struggle to understand cyclical human behavior from a flat counter, I engineered an `Hour_of_Day` feature. This allowed the model to recognize high-risk temporal patterns, such as sudden spikes in activity at 3:00 AM.
    """)
    
    # Phase 2
    st.subheader("Phase 2: The Machine Learning Core")
    st.markdown("""
    At the center of the engine sits an **XGBoost Classifier**, chosen for its high performance on structured, tabular data and its ability to handle severe class imbalance (where legitimate transactions outnumber fraud by massive margins).
    
    * **Asymmetric Optimization (F2 Score):** In the business world, all errors are not equal. Missing a \$5,000 fraudulent transaction (False Negative) is exponentially more damaging than accidentally blocking a \$5 coffee (False Positive). Instead of using standard accuracy, the model was optimized using the **F2 Score**, which weighs Recall higher than Precision. 
    * **Custom Decision Threshold:** By mathematical tuning, the default 50% probability threshold was lowered to **46.48%**. This ensures the AI prioritizes catching criminals, accepting a mathematically calculated, negligible increase in customer friction.
    """)
    
    # Phase 3
    st.subheader("Phase 3: The Business Rules Engine")
    st.markdown("""
    Machine learning models can act as "black boxes" and are difficult to update rapidly during a sudden, coordinated attack. To make the system production-ready, I built a **YAML-based Business Rules Engine** that operates sequentially after the ML scoring layer.
    
    This allows stakeholders to instantly override the AI without altering the underlying Python codebase or waiting for a model retrain. The active rules include:
    * **The Absolute Limit:** Instantly blocks any transaction over \$15,000 as a pure financial safety net.
    * **The Night-Owl Check:** Flags transactions over \$3,000 occurring between 1:00 AM and 5:00 AM for human review.
    * **Micro-Transaction Monitoring:** Places accounts on a watchlist if they attempt transactions under \$1.50, a common tactic used by criminals to test stolen cards before a larger attack.
    """)
    
    # Phase 4
    st.subheader("Phase 4: Simulation & Deployment")
    st.markdown("""
    To prove the engine's viability, the final application was designed to simulate a live, real-time data feed.
    
    * **Zero Data Leakage:** The engine is evaluated against a strictly isolated holdout stream of 5,000 transactions that the model has never seen before.
    * **Batch Processing Engine:** The UI allows users to inject 1,000-row chunks of streaming data into the system, dynamically applying the `Hour_of_Day` engineering, the StandardScaler, the XGBoost probabilities, and the YAML business rules in milliseconds.
    * **Custom UI Architecture:** The dashboard abandons standard UI components for a custom "Antigravity" CSS framework, utilizing soft-square geometry, Nunito typography, and highly diffused shadows to create a frictionless, enterprise-grade user experience.
    """)
    
    st.divider()
    
    st.info("**The Tech Stack**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data & AI**\n* Python\n* Pandas & NumPy\n* XGBoost\n* Scikit-Learn")
    with col2:
        st.markdown("**Architecture**\n* Joblib (Artifact Serialization)\n* PyYAML (Rules Engine)\n* Streamlit")
    with col3:
        st.markdown("**Frontend UI**\n* HTML5 / CSS3\n* Custom CSS Injection\n* Google Fonts (Nunito)")
        
    st.divider()
    st.markdown("<div style='text-align: center; opacity: 0.7;'>Designed and Engineered by <a href='https://github.com/dibyanshusamal9' target='_blank'>Dibyanshu Samal</a></div>", unsafe_allow_html=True)

elif page == "Fraud Scoring Engine":
    st.title("Transaction Fraud Risk & Rule Engine")
    st.markdown("Detect fraudulent transactions in real-time using a combination of machine learning and business rules.")
    st.divider()
    
    # Batch selection
    batch_option = st.selectbox(
        "Select Transaction Batch to Analyze",
        [
            "Batch 1 (Rows 1-1000)",
            "Batch 2 (Rows 1001-2000)",
            "Batch 3 (Rows 2001-3000)",
            "Batch 4 (Rows 3001-4000)",
            "Batch 5 (Rows 4001-5000)"
        ]
    )
    
    # Slice data
    batch_map = {
        "Batch 1 (Rows 1-1000)": (0, 1000),
        "Batch 2 (Rows 1001-2000)": (1000, 2000),
        "Batch 3 (Rows 2001-3000)": (2000, 3000),
        "Batch 4 (Rows 3001-4000)": (3000, 4000),
        "Batch 5 (Rows 4001-5000)": (4000, 5000)
    }
    
    start_idx, end_idx = batch_map[batch_option]
    df = df.iloc[start_idx:end_idx].copy()
    
    st.divider()
    
    # Recent transactions table
    st.subheader("Recent Transactions")
    st.caption("Confused by the V1-V28 columns? Check the **Dataset Overview** tab in the sidebar for the data dictionary.")
    st.dataframe(df.head(5).rename(columns=lambda x: x.replace('_', ' ')).rename(columns={'Time': 'Time (in s)'})
                 .style.set_properties(**{'text-align': 'left'})
                 .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                 .format({'Amount': '${:.2f}'}))
    
    # Analysis engine
    if st.button("Analyze Transactions", type="primary"):
        # Validate inputs
        required_cols = ['Time', 'Amount', 'Class']
        if not all(col in df.columns for col in required_cols):
            st.error(f"The dataset is missing required columns. Please ensure your data includes: {required_cols}.")
            st.stop()
            
        if len(df) == 0:
            st.warning("The transaction batch is empty. Please select a batch that contains data.")
            st.stop()

        with st.spinner("Analyzing transactions... this usually takes just a moment."):
            
            # Feature engineering
            X_features = df.drop('Class', axis=1)
            X_features['Hour_of_Day'] = (X_features['Time'] // 3600) % 24
            X_features['scaled_amount'] = scaler.transform(X_features['Amount'].values.reshape(-1, 1))
            X_features = X_features.drop(['Time', 'Amount'], axis=1)
            
            # Model predictions
            probabilities = model.predict_proba(X_features)[:, 1]
            ml_threshold = rules['global_system_rules']['ml_fraud_threshold']
            
            # Apply rules
            results = df.copy()
            results['ML_Probability'] = probabilities
            results['Final_Decision'] = "APPROVED" # Default
            results['Flag_Reason'] = "Clean"
            
            # Apply logic
            amount = results['Amount']
            hour = (results['Time'] // 3600) % 24
            ml_prob = results['ML_Probability']
            
            # Evaluate rules
            m1_hard_limit = amount >= rules['global_system_rules']['hard_amount_limit']
            
            night_owl = rules['custom_risk_scenarios']['night_owl_heavy_hitter']
            m2_night_owl = night_owl['active'] & (hour >= 1) & (hour <= 5) & (amount >= night_owl['condition']['min_amount'])
            
            micro = rules['custom_risk_scenarios']['micro_transaction_testing']
            m3_micro = micro['active'] & (amount <= micro['condition']['max_amount'])
            
            m4_ml = ml_prob >= ml_threshold
            
            condlist = [m1_hard_limit, m2_night_owl, m3_micro, m4_ml]
            
            results['Final_Decision'] = np.select(
                condlist, 
                ["BLOCKED", "MONITOR", "MONITOR", "BLOCKED"], 
                default="APPROVED"
            )
            
            results['Flag_Reason'] = np.select(
                condlist,
                [
                    "Rule: Hard Amount Limit Exceeded",
                    "Rule: High-Value Night Transaction",
                    "Rule: Micro-Transaction Testing",
                    "ML Score Exceeds Threshold"
                ],
                default="Clean"
            )
    
        # Display results
            # Precompute styling
            def color_status(val):
                if val == "BLOCKED": return 'background-color: rgba(225, 29, 72, 0.15)' # Rose
                if val == "MONITOR": return 'background-color: rgba(245, 158, 11, 0.15)' # Amber
                return ''

            styled_df = (results[['Amount', 'Time', 'ML_Probability', 'Final_Decision', 'Flag_Reason', 'Class']]
                         .rename(columns=lambda x: x.replace('_', ' '))
                         .rename(columns={'Time': 'Time (in s)'})
                         .sort_values(by='ML Probability', ascending=False)
                         .head(500) # Limit rows for performance
                         .style.set_properties(**{'text-align': 'left'})
                         .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
                         .map(color_status, subset=['Final Decision'])
                         .format({'Amount': '${:.2f}', 'Time (in s)': '{:.0f}', 'ML Probability': '{:.2%}'}))
                         
            st.divider()
            
            # Show summary banner
            total_blocks = len(results[results['Final_Decision'] == 'BLOCKED'])
            total_flags = len(results[results['Final_Decision'] == 'MONITOR'])
            
            if total_blocks == 0 and total_flags == 0:
                st.success("Analysis complete. No threats detected in this batch.")
            elif total_blocks > 0:
                st.error(f"High-risk activity detected. {total_blocks} transactions were automatically blocked.")
            else:
                st.warning(f"Suspicious activity detected. {total_flags} transactions were placed on the monitor list.")
            
            # Metrics summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Transactions Analyzed", len(results), help="Total number of transactions processed in this batch.")
            col2.metric("Approved", len(results[results['Final_Decision'] == 'APPROVED']), help="Transactions that safely passed all AI and business rule checks.")
            col3.metric("Rule Flags", len(results[results['Flag_Reason'].str.contains('Rule')]), help="Transactions intercepted by explicit business rules (e.g., high-value limits).")
            col4.metric("AI Blocks", len(results[results['Flag_Reason'] == 'ML Score Exceeds Threshold']), help="Transactions blocked solely because the AI model flagged them as high-risk.")
            
            st.divider()
            
            st.subheader("Analysis Results")
            st.caption("Curious about the exact thresholds? Check the **Business Rules** tab in the sidebar to understand how these decisions were made.")
            st.dataframe(
                styled_df, 
                height=400,
                column_config={
                    "ML Probability": st.column_config.Column(
                        help="The mathematical probability (0% to 100%) determined by our AI model that this transaction is fraudulent."
                    ),
                    "Final Decision": st.column_config.Column(
                        help="The ultimate system action taken. Transactions can be APPROVED, MONITOR (requires human oversight), or BLOCKED."
                    ),
                    "Flag Reason": st.column_config.Column(
                        help="The specific security rule or AI threshold that triggered the final decision."
                    ),
                    "Class": st.column_config.Column(
                        help="The actual historical ground-truth label. 0 means Legitimate, 1 means confirmed Fraud."
                    )
                }
            )

            # Explainable AI (XAI)
            st.divider()
            
            # Find highest risk transaction
            flagged_txns = results[results['Final_Decision'] != 'APPROVED']
            
            if len(flagged_txns) > 0:
                top_threat = flagged_txns.sort_values(by='ML_Probability', ascending=False).iloc[0]
                
                st.subheader(f"Explainable AI: Threat Deep Dive")
                
                col_xai1, col_xai2 = st.columns([1, 2])
                
                with col_xai1:
                    st.markdown("#### Threat Profile")
                    st.markdown(f"**Target ID:** `{top_threat.name}`")
                    st.markdown(f"**Amount:** `${top_threat['Amount']:.2f}`")
                    st.markdown(f"**System Action:** `{top_threat['Final_Decision']}`")
                    st.markdown(f"**Primary Trigger:** *{top_threat['Flag_Reason']}*")
                    
                    st.divider()
                    st.metric("AI Fraud Probability", f"{top_threat['ML_Probability']:.2%}")
                
                with col_xai2:
                    st.markdown(f"**Anomaly Breakdown (Row {top_threat.name})**")
                    
                    # Calculate deviation from batch average
                    features_to_analyze = ['Amount', 'Time', 'V3', 'V4', 'V10', 'V11', 'V12', 'V14']
                    means = results[features_to_analyze].mean()
                    stds = results[features_to_analyze].std()
                    
                    # Z-Score calculation
                    deviations = ((top_threat[features_to_analyze] - means) / stds).fillna(0)
                    
                    # Render deviation chart
                    st.bar_chart(deviations, color="#e11d48")
                    st.caption("Bars stretching far above or below the zero-line indicate highly abnormal behavior compared to the batch average. (Measured in Standard Deviations)")
            else:
                st.info("No high-risk transactions detected in this batch to analyze.")

elif page == "Business Rules":
    st.title("Business Rules")
    st.markdown("This page outlines the core business logic and rule-based thresholds that run in parallel with the machine learning model. These rules act as explicit safety nets and risk triggers.")
    st.divider()
    
    st.subheader("1. The Machine Learning Baseline")
    st.markdown('''
**The Rule:**  
Any transaction that the machine learning model scores above a **46.48% probability** of being fraud is automatically blocked.

**The Business Logic:**  
Instead of using a standard 50% threshold, we mathematically optimized this number to prioritize catching criminals. By slightly lowering the threshold, we accept a tiny increase in false alarms (annoying a few customers) to ensure we catch a significantly higher percentage of actual fraud.
''')
    
    st.divider()
    
    st.subheader("2. The Absolute Amount Limit")
    st.markdown('''
**The Rule:**  
Any transaction equal to or greater than **$15,000** is instantly blocked, regardless of what the machine learning model predicts.

**The Business Logic:**  
This is a pure financial safety net. Even if a transaction looks perfectly normal to the algorithm, the financial risk of a single $15,000 loss is too high to process automatically.
''')
    
    st.divider()
    
    st.subheader('3. The "Night Owl" High-Value Check')
    st.markdown('''
**The Rule:**  
If a transaction occurs between **1:00 AM and 5:00 AM** AND the amount is **$3,000 or more**, the system changes its status to "Monitor."

**The Business Logic:**  
Fraudsters often operate in the middle of the night when customers are asleep and unable to receive mobile alert notifications. While we don't want to automatically block legitimate night-shift workers or travelers making large purchases, this rule ensures these risky transactions are put in a queue for a human analyst to double-check.
''')
    
    st.divider()
    
    st.subheader("4. The Micro-Transaction Monitor")
    st.markdown('''
**The Rule:**  
Any transaction of **$1.50 or less** is flagged with a "Monitor" status.

**The Business Logic:**  
Before criminals make a massive purchase with a stolen credit card, they frequently "test" the card by buying something incredibly cheap—like a $1 digital download or a parking meter fee—just to see if the card is active. This rule doesn't block the transaction, but it puts the account on a watchlist so the system is highly sensitive to whatever that user does next.
''')

elif page == "Dataset Overview":
    st.title("Dataset Overview & Dictionary")
    st.markdown("Understanding the data is the first step to trusting the model. This page outlines the exact data points our engine analyzes in real-time.")
    st.divider()
    
    st.subheader("The Data Dictionary")
    st.markdown("""
    Our system analyzes a continuous stream of credit card transactions. To protect user privacy, most personal identifiable information (PII) has been removed, leaving only mathematical representations of the user's behavior alongside core transaction details.
    
    * **Time:** The number of seconds elapsed between this transaction and the very first transaction in the dataset. We use this to calculate the 'Hour of Day' to spot suspicious late-night activity.
    * **Amount:** The total transaction value in US Dollars ($).
    * **V1 through V28:** These are anonymized security variables. They represent complex behavioral patterns (like location data, device type, or past purchase history) that have been transformed using a mathematical process called PCA (Principal Component Analysis) to ensure total privacy.
    * **Class:** The actual ground-truth label. `0` means the transaction was legitimate, and `1` means the transaction was confirmed fraud.
    """)
    
    st.divider()
    
    st.subheader("Real-World Examples")
    st.markdown("Here is what the raw data looks like before the model processes it:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Example A: The Everyday Purchase**")
        st.markdown("""
        **Scenario:** Buying a coffee on the way to work.
        * **Time:** `28500` (Approx 7:55 AM)
        * **Amount:** `$4.50`
        * **Class:** `0` (Legitimate)
        * **V-Features:** Normal baseline values indicating a recognized device at a frequent location.
        """)
        
    with col2:
        st.error("**Example B: The Stolen Card Attack**")
        st.markdown("""
        **Scenario:** A fraudster testing a stolen card online at 3 AM.
        * **Time:** `10800` (Approx 3:00 AM)
        * **Amount:** `$0.99` (Micro-transaction test)
        * **Class:** `1` (Fraud)
        * **V-Features:** Highly erratic values indicating an unknown IP address and a mismatched shipping location.
        """)
