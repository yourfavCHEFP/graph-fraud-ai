"""
Graph Fraud AI -- Hugging Face Spaces Production Demo.

Architecture:
1. GitHub (Research): Full ML pipeline, 10 notebooks, model training.
2. Hugging Face (Production): Streamlit UI + In-process GraphSAGE inference.

This app implements the "Bridge" architecture where the production demo 
runs entirely in-process for stability and performance on HF Spaces.
"""

import time

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import numpy as np
from src.inference.predictor import FraudPredictor

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Graph-Fraud AI | Production Demo",
    page_icon="🛡️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #0068c9;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        color: white;
    }
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🛡️ Graph-Fraud AI")
st.sidebar.markdown("### Production Inference Engine")
st.sidebar.divider()

@st.cache_resource(show_spinner="Initializing GraphSAGE Model & Transaction Graph...")
def load_predictor():
    try:
        return FraudPredictor()
    except Exception as e:
        return str(e)

predictor = load_predictor()

if isinstance(predictor, str):
    st.sidebar.error(f"Initialization Failed: {predictor}")
    st.stop()
else:
    num_nodes = predictor.num_nodes
    st.sidebar.success(f"Model Ready: {predictor.model_name}")
    st.sidebar.info(f"Graph Scale: {num_nodes:,} nodes")
    st.sidebar.divider()
    st.sidebar.markdown("""
    **Architecture:**
    - GitHub: Research & Training
    - HF Space: Real-time Inference
    - Model: GraphSAGE (PyG)
    """)

# --- MAIN UI ---
def main():
    st.title("🛡️ Transaction Fraud Analysis")
    st.markdown("Enter a Transaction ID to trigger the **GraphSAGE** inference pipeline and visualize relational fraud risk.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("Input Transaction")
        transaction_id = st.number_input(
            "Transaction ID (Index)", 
            min_value=0, 
            max_value=num_nodes - 1, 
            value=0, 
            help="The numerical index of the transaction in the global graph."
        )
        
        analyze_btn = st.button("RUN INFERENCE")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Metadata Card
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("Pipeline Details")
        st.json({
            "Stage 1": "Startup Graph Initialization",
            "Stage 2": "Full-Graph GraphSAGE Scoring (once, at startup)",
            "Stage 3": "Transaction Probability Lookup (per request)",
            "Stage 4": "Local Neighborhood Context (per request)",
            "Status": "Healthy"
        })
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if analyze_btn:
            with st.spinner("Processing relational patterns..."):
                try:
                    start_time = time.perf_counter()
                    result = predictor.predict_transaction(int(transaction_id))
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Result Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    color = "#e74c3c" if result["prediction"] == "fraud" else "#2ecc71"
                    
                    m_col1.markdown(f'<p class="metric-label">Decision</p><p class="metric-value" style="color:{color}">{result["prediction"].upper()}</p>', unsafe_allow_html=True)
                    m_col2.markdown(f'<p class="metric-label">Fraud Probability</p><p class="metric-value">{result["fraud_probability"]:.2%}</p>', unsafe_allow_html=True)
                    m_col3.markdown(f'<p class="metric-label">Lookup Time</p><p class="metric-value">{elapsed_ms:.1f}ms</p>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Explanation Section
                    st.subheader("🔍 Explainability Analysis")
                    explanation = result["explanation"]
                    
                    if result["prediction"] == "fraud":
                        st.warning(f"**Risk Level: {explanation['risk_level']}**")
                    else:
                        st.success(f"**Risk Level: {explanation['risk_level']}**")
                        
                    st.markdown("**Relational Risk Factors:**")
                    for factor in explanation["risk_factors"]:
                        st.markdown(f"- {factor}")
                        
                    st.divider()
                    
                    # Graph Visualization
                    st.subheader("🕸️ Relational Context (Local Subgraph)")
                    graph_context = explanation["graph_context"]
                    neighbors = graph_context["neighbors"]
                    
                    if neighbors:
                        G = nx.Graph()
                        # Central Node
                        G.add_node(transaction_id, label=f"TX {transaction_id}", type='target')
                        # Neighbors
                        for n in neighbors:
                            G.add_edge(transaction_id, n)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pos = nx.spring_layout(G, seed=42)
                        
                        node_colors = [
                            "#e74c3c" if node == transaction_id else "#3498db"
                            for node in G.nodes()
                        ]
                        
                        nx.draw(
                            G, pos, ax=ax, with_labels=True, 
                            node_color=node_colors, node_size=1200, 
                            font_size=9, font_color="white", 
                            font_weight='bold', edge_color='#ced4da'
                        )
                        
                        ax.set_title(f"Transaction {transaction_id} Relational Neighborhood", fontsize=12, fontweight='bold')
                        st.pyplot(fig)
                    else:
                        st.info("No immediate relational neighbors found for this node.")
                        
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
        else:
            # Placeholder State
            st.info("👈 Enter a Transaction ID and click 'Run Inference' to see the graph analysis.")

if __name__ == "__main__":
    main()
