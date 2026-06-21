"""
Shopper Spectrum — Streamlit App
=================================
Two modules:
  1. Product Recommendation  →  item-based collaborative filtering
  2. Customer Segmentation   →  RFM-based KMeans cluster prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 1rem; }

    /* Cards */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1976D2;
        font-size: 1rem;
        color: #1a1a2e;
    }
    .card-rank { font-weight: 700; color: #1976D2; margin-right: 6px; }

    /* Segment badge */
    .badge {
        display: inline-block;
        padding: 0.45rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin-top: 0.6rem;
    }
    .badge-high-value  { background: #1565C0; }
    .badge-regular     { background: #2E7D32; }
    .badge-occasional  { background: #E65100; }
    .badge-at-risk     { background: #B71C1C; }

    /* Metric cards */
    .metric-row {
        display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1; min-width: 140px;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-top: 4px solid #1976D2;
    }
    .metric-label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #1a1a2e; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, 'kmeans_model.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_map.pkl'), 'rb') as f:
        label_map = pickle.load(f)
    sim_df = pd.read_pickle(os.path.join(MODEL_DIR, 'similarity_matrix.pkl'))
    with open(os.path.join(MODEL_DIR, 'product_list.pkl'), 'rb') as f:
        product_list = pickle.load(f)
    return kmeans, scaler, label_map, sim_df, product_list

try:
    kmeans, scaler, label_map, sim_df, product_list = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("*E-Commerce Analytics Platform*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📦 Product Recommendations", "👥 Customer Segmentation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        """
        **Model:** KMeans (k=4) + Cosine Similarity  
       
        """
    )


# ── Helper functions ──────────────────────────────────────────────────────────
def get_recommendations(product_name: str, top_n: int = 5):
    query = product_name.upper().strip()
    if query in sim_df.index:
        matched = query
    else:
        matches = [p for p in sim_df.index if query in p]
        if not matches:
            return [], None
        matched = matches[0]
    recs = (
        sim_df[matched]
        .drop(matched)
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return recs, matched


def predict_segment(recency, frequency, monetary):
    x = np.array([[recency, frequency, monetary]])
    x_scaled = scaler.transform(x)
    cluster_id = kmeans.predict(x_scaled)[0]
    return label_map[cluster_id], cluster_id


SEGMENT_META = {
    'High-Value': {
        'color': '#1565C0', 'badge': 'badge-high-value', 'icon': '💎',
        'desc': 'Recent, frequent, and big spenders. Your most loyal and profitable customers.',
        'actions': ['Reward with loyalty points', 'Early access to new products', 'VIP-only offers']
    },
    'Regular': {
        'color': '#2E7D32', 'badge': 'badge-regular', 'icon': '✅',
        'desc': 'Steady purchasers who engage regularly but haven\'t hit premium levels yet.',
        'actions': ['Upsell with bundle deals', 'Send personalised recommendations', 'Invite to loyalty programme']
    },
    'Occasional': {
        'color': '#E65100', 'badge': 'badge-occasional', 'icon': '🕐',
        'desc': 'Infrequent buyers who purchase occasionally. High growth potential.',
        'actions': ['Trigger re-engagement emails', 'Seasonal promotions', 'Flash sale alerts']
    },
    'At-Risk': {
        'color': '#B71C1C', 'badge': 'badge-at-risk', 'icon': '⚠️',
        'desc': 'Used to purchase but haven\'t been active recently. Act fast before they churn.',
        'actions': ['Win-back campaign with discount', 'Ask for feedback', 'Show new arrivals they might like']
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🛒 Shopper Spectrum")
    st.markdown("### Customer Segmentation & Product Recommendations in E-Commerce")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### What this app does
        
        This platform analyses transactional data from a UK-based online retail store
        to surface two core insights:
        
        **📦 Product Recommendations**  
        Enter any product name and instantly get 5 similar products a customer is likely 
        to buy — powered by item-based collaborative filtering with cosine similarity.
        
        **👥 Customer Segmentation**  
        Input a customer's Recency, Frequency, and Monetary (RFM) values to predict 
        which segment they fall into — so your marketing team knows exactly how to engage them.
        """)

    with col2:
        st.markdown("#### Segment Guide")
        for seg, meta in SEGMENT_META.items():
            st.markdown(f"""
            <div style="background:#f8f9fa; border-radius:10px; padding:0.8rem 1rem; 
                        margin-bottom:0.5rem; border-left:4px solid {meta['color']};">
                <strong>{meta['icon']} {seg}</strong><br>
                <span style="font-size:0.85rem; color:#555;">{meta['desc']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:#e8f5e9; border-radius:10px; padding:1rem 1.5rem; font-size:0.9rem;">
    <strong>📊 Dataset:</strong> UCI Online Retail II — ~1M real UK e-commerce transactions (2009–2011)<br>
    <strong>🔗 Source:</strong> <a href="https://archive.ics.uci.edu/dataset/502/online+retail+ii" target="_blank">
    archive.ics.uci.edu/dataset/502/online+retail+ii</a> | 
    <a href="https://www.kaggle.com/datasets/mathchi/online-retail-ii-data-set-from-ml-repository" target="_blank">Kaggle Mirror</a>
    </div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.warning(
            f"⚠️ Model files not found. Please run the Jupyter notebook first to generate "
            f"`/models/*.pkl` files.\n\nError: `{load_error}`"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product Recommendations":
    st.markdown("## 📦 Product Recommender")
    st.markdown("Type a product name to get 5 similar items customers also buy.")
    st.markdown("---")

    if not models_loaded:
        st.error(f"Models not loaded. Run the notebook first.\nError: `{load_error}`")
        st.stop()

    col_input, col_info = st.columns([2, 1])

    with col_input:
        product_input = st.text_input(
            "Enter Product Name",
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
            help="Partial matches are supported — just type part of the product name."
        )

        recommend_btn = st.button("🔍 Get Recommendations", type="primary")

    with col_info:
        st.markdown(f"""
        <div style="background:#e3f2fd; border-radius:10px; padding:1rem; font-size:0.85rem; margin-top:1.5rem;">
        <strong>ℹ️ How it works</strong><br><br>
        Each product is represented as a vector of customer purchase quantities.
        <strong>Cosine similarity</strong> measures the angle between two product vectors —
        products frequently bought by the same customers score higher.
        </div>
        """, unsafe_allow_html=True)

    if recommend_btn:
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            with st.spinner("Finding similar products..."):
                recs, matched = get_recommendations(product_input.strip())

            if not recs:
                st.error(f"❌ No match found for **\"{product_input}\"**. Try a different keyword.")
            else:
                if matched.upper() != product_input.upper().strip():
                    st.info(f"💡 Matched to: **{matched}**")

                st.markdown(f"#### 🎯 Recommended Products for: *{matched}*")
                st.markdown("")

                for i, rec in enumerate(recs, 1):
                    sim_score = sim_df.loc[matched, rec] if matched in sim_df.index else 0
                    st.markdown(f"""
                    <div class="card">
                        <span class="card-rank">#{i}</span>
                        {rec}
                        <span style="float:right; font-size:0.8rem; color:#888;">
                            similarity: {sim_score:.3f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔎 Browse Available Products")
    search_term = st.text_input("Filter product list", placeholder="Type to search...")
    if search_term:
        filtered = [p for p in product_list if search_term.upper() in p.upper()][:50]
        if filtered:
            st.write(f"Showing {len(filtered)} results:")
            cols = st.columns(2)
            for i, p in enumerate(filtered):
                cols[i % 2].markdown(f"• {p}")
        else:
            st.info("No products matched your search.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.markdown("## 👥 Customer Segmentation")
    st.markdown("Enter a customer's RFM values to predict their segment.")
    st.markdown("---")

    if not models_loaded:
        st.error(f"Models not loaded. Run the notebook first.\nError: `{load_error}`")
        st.stop()

    col_form, col_guide = st.columns([1, 1])

    with col_form:
        st.markdown("#### Customer RFM Input")

        recency   = st.number_input("📅 Recency (days since last purchase)",
                                    min_value=0, max_value=730, value=30, step=1)
        frequency = st.number_input("🔁 Frequency (number of transactions)",
                                    min_value=1, max_value=500, value=5, step=1)
        monetary  = st.number_input("💰 Monetary (total spend in £)",
                                    min_value=0.0, max_value=100000.0, value=500.0, step=50.0)

        predict_btn = st.button("🎯 Predict Segment", type="primary", use_container_width=True)

    with col_guide:
        st.markdown("#### What is RFM?")
        st.markdown("""
        | Metric | Meaning | Low = ? | High = ? |
        |---|---|---|---|
        | **Recency** | Days since last purchase | Active buyer | Dormant |
        | **Frequency** | Total # of orders | Occasional | Loyal |
        | **Monetary** | Total £ spent | Low spender | Big spender |
        """)
        st.markdown("""
        <div style="background:#fff8e1; border-radius:10px; padding:0.8rem 1rem; font-size:0.85rem; margin-top:0.5rem;">
        💡 <strong>Tip:</strong> Low recency means the customer bought recently (good!). 
        High frequency and monetary values indicate a more engaged customer.
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Predicting segment..."):
            segment, cluster_id = predict_segment(recency, frequency, monetary)
            meta = SEGMENT_META[segment]

        st.markdown("---")
        st.markdown("#### Prediction Result")

        st.markdown(f"""
        <div style="background:#f8f9fa; border-radius:14px; padding:1.5rem 2rem; 
                    border-left:6px solid {meta['color']}; margin-bottom:1rem;">
            <div style="font-size:0.85rem; color:#888; margin-bottom:0.3rem;">PREDICTED SEGMENT</div>
            <div style="font-size:2rem; font-weight:800; color:{meta['color']};">
                {meta['icon']} {segment}
            </div>
            <div style="font-size:0.95rem; color:#444; margin-top:0.6rem;">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Recency</div>
                <div class="metric-value">{recency}d</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Frequency</div>
                <div class="metric-value">{frequency}</div>
            </div>
            <div class="metric-card" style="border-top-color:#388E3C;">
                <div class="metric-label">Monetary</div>
                <div class="metric-value">£{monetary:,.0f}</div>
            </div>
            <div class="metric-card" style="border-top-color:{meta['color']};">
                <div class="metric-label">Cluster ID</div>
                <div class="metric-value">{cluster_id}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommended actions
        st.markdown("#### 📋 Recommended Actions")
        for action in meta['actions']:
            st.markdown(f"""
            <div class="card" style="border-left-color:{meta['color']};">
                ✅ {action}
            </div>
            """, unsafe_allow_html=True)

    # Segment reference at the bottom
    with st.expander("📊 View All Segment Profiles"):
        for seg, meta in SEGMENT_META.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{meta['icon']} {seg}**")
                st.markdown(f"<span style='color:{meta['color']}; font-weight:600;'>●</span> {meta['desc']}",
                            unsafe_allow_html=True)
            with col2:
                for a in meta['actions']:
                    st.markdown(f"- {a}")
            st.markdown("---")
