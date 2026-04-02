import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Transformer Observation Deck", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("results_all.csv")
    m = df["model"].str.extract(r"^metrics_([^_]+)_([^_]+)_(\d+)")
    df["family"] = m[0]
    df["size"] = m[1]
    df["resolution"] = m[2].astype(int)
    return df

df = load_data()

st.title("\U0001f52d Transformer Observation Deck")
st.markdown("Interactive benchmark explorer for Vision Transformer variants. Filter by model family and resolution to compare inference efficiency and segmentation quality.")

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filters")
    families = sorted(df["family"].dropna().unique())
    selected_families = st.multiselect("Model families", families, default=families)
    resolutions = sorted(df["resolution"].unique())
    selected_resolutions = st.multiselect("Resolutions", resolutions, default=resolutions)

df_f = df[df["family"].isin(selected_families) & df["resolution"].isin(selected_resolutions)]

tab_eff, tab_seg = st.tabs(["\u26a1 Inference Efficiency", "\U0001f3af Segmentation Quality vs Efficiency"])

# ─────────────────────────────────────────────
# TAB 1: Inference Efficiency
# ─────────────────────────────────────────────
with tab_eff:

    # ① Parameters vs Throughput @ 224
    st.subheader("\u2460 Parameters vs Throughput @ 224")
    df_224 = df_f[df_f["resolution"] == 224]
    if not df_224.empty:
        fig1 = px.scatter(
            df_224, x="inferency/number of parameters", y="inferency/throughput/value",
            color="family", hover_data=["model", "pre-acc"],
            labels={"inferency/number of parameters": "Parameters",
                    "inferency/throughput/value": "Throughput (img/s)"},
            title="Parameters vs Throughput @ 224"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data for 224px with current filters.")

    col1, col2 = st.columns(2)

    with col1:
        # ② FLOPs Across Resolutions
        st.subheader("\u2461 FLOPs Across Resolutions")
        fig2 = px.line(
            df_f.sort_values("resolution"), x="resolution", y="inferency/flops",
            color="family", markers=True, hover_data=["model"],
            labels={"inferency/flops": "FLOPs", "resolution": "Resolution"},
            title="FLOPs Across Resolutions"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # ③ Throughput Degradation with Resolution
        st.subheader("\u2462 Throughput Degradation with Resolution")
        base = (
            df_f[df_f["resolution"] == df_f["resolution"].min()]
            .groupby(["family", "size"])["inferency/throughput/value"].max()
        )
        df_rr = df_f.copy()
        df_rr["thr_vs_min"] = df_rr.apply(
            lambda r: r["inferency/throughput/value"] /
                      base.get((r["family"], r["size"]), r["inferency/throughput/value"]),
            axis=1
        )
        fig3 = px.line(
            df_rr.sort_values("resolution"), x="resolution", y="thr_vs_min",
            color="family", markers=True, hover_data=["model"],
            labels={"thr_vs_min": "% Throughput (vs min res)", "resolution": "Resolution"},
            title="Throughput Degradation with Resolution"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ④ Inference Memory @ 224
    st.subheader("\u2463 Inference Memory @ 224 (batch = 1)")
    if not df_224.empty:
        fig4 = px.bar(
            df_224.sort_values("inferency/inference_memory_@1"),
            x="model", y="inferency/inference_memory_@1",
            color="family",
            labels={"inferency/inference_memory_@1": "Memory (bytes)", "model": "Model"},
            title="Inference Memory @ 224"
        )
        fig4.update_xaxes(tickangle=60)
        st.plotly_chart(fig4, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # ⑤ Efficiency Frontier — Speed vs Memory
        st.subheader("\u2464 Efficiency Frontier \u2014 Speed vs Memory")
        fig5 = px.scatter(
            df_f, x="inferency/inference_memory_@1", y="inferency/throughput/value",
            color="family", hover_data=["model", "resolution"],
            labels={"inferency/inference_memory_@1": "Memory (bytes)",
                    "inferency/throughput/value": "Throughput (img/s)"},
            title="Efficiency Frontier \u2014 Speed vs Memory"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col4:
        # ⑥ Resolution Robustness — % Throughput Kept
        st.subheader("\u2465 Resolution Robustness \u2014 % Throughput Kept")
        base_224 = (
            df_f[df_f["resolution"] == 224]
            .groupby(["family", "size"])["inferency/throughput/value"].max()
        )
        df_rob = df_f.copy()
        df_rob["thr_vs_224"] = df_rob.apply(
            lambda r: r["inferency/throughput/value"] /
                      base_224.get((r["family"], r["size"]), r["inferency/throughput/value"]),
            axis=1
        )
        fig6 = px.line(
            df_rob.sort_values("resolution"), x="resolution", y="thr_vs_224",
            color="family", line_group="size", markers=True, hover_data=["model"],
            labels={"thr_vs_224": "% Throughput vs 224", "resolution": "Resolution"},
            title="Resolution Robustness \u2014 % Throughput Kept"
        )
        st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2: Segmentation Quality vs Efficiency
# ─────────────────────────────────────────────
with tab_seg:

    # ⑦ Pretrain Accuracy vs Best IoU
    st.subheader("\u2466 Pretrain Accuracy vs Best IoU")
    best_iou = df_f.groupby("model")["iou"].max().reset_index()
    merged = best_iou.merge(df_f[["model", "pre-acc", "family"]].drop_duplicates("model"), on="model")
    fig7 = px.scatter(
        merged, x="pre-acc", y="iou", color="family", hover_data=["model"],
        labels={"pre-acc": "Pretrain Accuracy", "iou": "Best IoU"},
        title="Pretrain Accuracy vs Best IoU"
    )
    st.plotly_chart(fig7, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        # ⑧ IoU vs Throughput
        st.subheader("\u2467 IoU vs Throughput")
        fig8 = px.scatter(
            df_f, x="inferency/throughput/value", y="iou",
            color="family", hover_data=["model", "resolution"],
            labels={"inferency/throughput/value": "Throughput (img/s)", "iou": "IoU"},
            title="IoU vs Throughput"
        )
        st.plotly_chart(fig8, use_container_width=True)

    with col6:
        # ⑨ IoU vs Memory
        st.subheader("\u2468 IoU vs Memory")
        fig9 = px.scatter(
            df_f, x="inferency/inference_memory_@1", y="iou",
            color="family", hover_data=["model", "resolution"],
            labels={"inferency/inference_memory_@1": "Memory (bytes)", "iou": "IoU"},
            title="IoU vs Memory"
        )
        st.plotly_chart(fig9, use_container_width=True)

    col7, col8 = st.columns(2)

    with col7:
        # ⑩ IoU vs FLOPs
        st.subheader("\u2469 IoU vs FLOPs")
        fig10 = px.scatter(
            df_f, x="inferency/flops", y="iou",
            color="family", hover_data=["model", "resolution"],
            labels={"inferency/flops": "FLOPs", "iou": "IoU"},
            title="IoU vs FLOPs"
        )
        st.plotly_chart(fig10, use_container_width=True)

    with col8:
        # ⑪ IoU vs Parameters
        st.subheader("\u246a IoU vs Parameters")
        fig11 = px.scatter(
            df_f, x="inferency/number of parameters", y="iou",
            color="family", hover_data=["model", "resolution"],
            labels={"inferency/number of parameters": "Parameters", "iou": "IoU"},
            title="IoU vs Parameters"
        )
        st.plotly_chart(fig11, use_container_width=True)
