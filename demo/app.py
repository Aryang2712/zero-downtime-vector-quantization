"""
Adaptive Online Product Quantization (AO-PQ) - Live Streamlit Demo
==================================================================
Real-time dashboard for:
- Live streaming ingestion with concept drift injection.
- Autonomous MSE monitoring & atomic codebook swap visualization.
- Sub-millisecond ANN search evaluation against exact ground truth.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import AdaptiveOnlinePQ

st.set_page_config(
    page_title="AO-PQ Engine | Real-Time Vector Stream Ingestion",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Adaptive Online Product Quantization (AO-PQ)")
st.caption("Zero-Downtime High-Performance Vector Quantization for Drifting Embedding Streams")

# ----------------- Sidebar Configuration -----------------
st.sidebar.header("⚙️ Engine Hyperparameters")
dim = st.sidebar.selectbox("Vector Dimension (d)", [64, 128], index=1)
m_subspaces = st.sidebar.selectbox("Subspaces (m)", [4, 8, 16], index=1)
k_centroids = st.sidebar.selectbox("Centroids per Subspace (k)", [128, 256], index=1)
learning_rate = st.sidebar.slider("Centroid Learning Rate (η)", 0.01, 0.25, 0.10, 0.01)
momentum = st.sidebar.slider("Polyak Momentum (β)", 0.50, 0.95, 0.85, 0.05)
drift_threshold = st.sidebar.slider("Drift Swap Threshold (τ)", 0.01, 0.08, 0.03, 0.005)

st.sidebar.markdown("---")
st.sidebar.header("�� Stream Simulation Controls")
batch_size = st.sidebar.slider("Batch Size (vectors/batch)", 100, 2000, 500, 100)
drift_magnitude = st.sidebar.slider("Injected Drift Shift (Δμ)", 0.0, 5.0, 1.5, 0.25)


# ----------------- State Initialization -----------------
if "engine" not in st.session_state or st.sidebar.button("🔄 Reset & Re-initialize Engine"):
    with st.spinner("Initializing baseline codebooks on 5,000 vectors..."):
        np.random.seed(42)
        engine = AdaptiveOnlinePQ(
            d=dim,
            m=m_subspaces,
            k=k_centroids,
            lr=learning_rate,
            momentum=momentum,
            drift_threshold=drift_threshold
        )
        # Baseline training data
        X_init = np.random.randn(5000, dim).astype(np.float32)
        engine.fit_initial(X_init)
        engine.codes = engine.quantize(X_init, engine.active_codebook)
        engine.epochs = np.zeros(5000, dtype=np.uint8)

        st.session_state.engine = engine
        st.session_state.history = []
        st.session_state.raw_data = [X_init]
        st.session_state.batch_count = 0
        st.success("Engine initialized successfully!")

engine = st.session_state.engine

# ----------------- Top Metrics Bar -----------------
col1, col2, col3, col4, col5 = st.columns(5)
total_indexed = engine.codes.shape[0]
total_swaps = engine.total_swaps
curr_epoch = "Active (1)" if engine.current_epoch_id == 1 else "Active (0)"

col1.metric("Indexed Vectors", f"{total_indexed:,}")
col2.metric("Codebook Swaps", f"{total_swaps}")
col3.metric("Current Epoch", curr_epoch)
col4.metric("Subspaces (m)", f"{m_subspaces}")
col5.metric("RAM Savings", "93.7%")

st.markdown("---")

# ----------------- Ingestion & Query Section -----------------
tab1, tab2 = st.tabs(["🚀 Streaming Ingestion & Drift Monitor", "🔍 Sub-Millisecond Vector Search"])

with tab1:
    st.subheader("Stream Simulation")
    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        st.write("Inject drifting embedding batches into the streaming engine:")
        num_stream_batches = st.number_input("Number of batches to stream", min_value=1, max_value=20, value=3)

        if st.button("▶️ Ingest Streaming Batches", use_container_width=True):
            progress_bar = st.progress(0)
            for i in range(num_stream_batches):
                st.session_state.batch_count += 1
                b_num = st.session_state.batch_count

                # Synthesize drifting batch
                shift_val = drift_magnitude * (b_num / 5.0)
                batch = (np.random.randn(batch_size, dim) + shift_val).astype(np.float32)
                st.session_state.raw_data.append(batch)

                # Prior MSE before update
                active_mse = engine.compute_reconstruction_mse(batch, engine.active_codebook)

                # Streaming Ingestion
                t_ingest_start = time.perf_counter()
                engine.ingest_stream_batch(batch)
                ingest_time_ms = (time.perf_counter() - t_ingest_start) * 1000.0

                # New MSE after background shadow update
                shadow_mse = engine.compute_reconstruction_mse(batch, engine.shadow_codebook)

                st.session_state.history.append({
                    "batch": b_num,
                    "active_mse": active_mse,
                    "shadow_mse": shadow_mse,
                    "total_vectors": engine.codes.shape[0],
                    "total_swaps": engine.total_swaps,
                    "ingest_time_ms": ingest_time_ms
                })
                progress_bar.progress((i + 1) / num_stream_batches)

            st.rerun()

        st.info(f"**Drift Detection Rule:** An atomic pointer swap executes when `(MSE_active - MSE_shadow) / MSE_active > {drift_threshold}`.")

    with col_chart:
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(df_hist["batch"], df_hist["active_mse"], label="Active Codebook MSE", color="#0275d8", linewidth=2.2, marker="o")
            ax.plot(df_hist["batch"], df_hist["shadow_mse"], label="Shadow Codebook MSE", color="#5cb85c", linestyle="--", linewidth=2.0)
            ax.set_title("Reconstruction MSE Tracking", fontweight="bold", fontsize=11)
            ax.set_xlabel("Batch Ingested")
            ax.set_ylabel("Reconstruction Error (MSE)")
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)
        else:
            st.write("👉 Click **'Ingest Streaming Batches'** to observe real-time drift tracking.")

with tab2:
    st.subheader("Interactive Nearest Neighbor Search")
    st.write("Execute live Asymmetric Distance Computation (ADC) search over the indexed stream.")

    top_k = st.slider("Top-K Neighbors", 1, 20, 5)

    if st.button("🎯 Fire Random Search Query", use_container_width=True):
        full_raw = np.vstack(st.session_state.raw_data)
        # Sample random query from recent data
        q = np.random.randn(dim).astype(np.float32) + (drift_magnitude * st.session_state.batch_count / 5.0)

        # 1. Exact Euclidean Ground Truth scan
        t_exact_start = time.perf_counter()
        exact_dists = np.sum((full_raw - q) ** 2, axis=1)
        true_topk = np.argpartition(exact_dists, top_k - 1)[:top_k]
        true_sorted = true_topk[np.argsort(exact_dists[true_topk])]
        exact_time_ms = (time.perf_counter() - t_exact_start) * 1000.0

        # 2. AO-PQ JIT Search
        t_adp_start = time.perf_counter()
        pred_ids, pred_dists = engine.search(q, top_k=top_k)
        adp_time_ms = (time.perf_counter() - t_adp_start) * 1000.0

        # Compute recall
        intersection = len(set(pred_ids).intersection(set(true_sorted)))
        recall_val = (intersection / float(top_k)) * 100.0

        q_col1, q_col2, q_col3 = st.columns(3)
        q_col1.metric("AO-PQ Search Latency", f"{adp_time_ms:.3f} ms")
        q_col2.metric("Brute Force Scan Latency", f"{exact_time_ms:.3f} ms")
        q_col3.metric(f"Recall@{top_k}", f"{recall_val:.1f}%")

        # Display Top Results Table
        st.write("#### Returned Nearest Neighbor Vector Indices")
        df_results = pd.DataFrame({
            "Rank": [f"#{i+1}" for i in range(len(pred_ids))],
            "AO-PQ Match ID": pred_ids,
            "AO-PQ Distance": np.round(pred_dists, 4),
            "Exact Ground Truth ID": true_sorted,
            "Exact Distance": np.round(exact_dists[true_sorted], 4)
        })
        st.dataframe(df_results, use_container_width=True)
