---
title: Transformer Observation Deck
emoji: 🔭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🔭 Transformer Observation Deck

Interactive benchmark dashboard for Vision Transformer variants.

Explore **inference efficiency** and **segmentation quality** across models, resolutions, and parameter counts — all in your browser, no install needed.

## Metrics covered

- Parameters, FLOPs, MACs, Throughput, Inference Memory
- IoU (segmentation), Pre-trained accuracy
- Resolution robustness and degradation

## Data

Based on `results_all.csv` — one row per (model, resolution) combination.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
