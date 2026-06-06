Full functional demo hosted [here] (https://huggingface.co/spaces/RSakib/MovieLens-Hybrid-SVD-Recommendation-GUI) on my HuggingFace

# 🎬 MovieLens Hybrid SVD Recommendations GUI

A full-stack, hybrid movie recommendation system built to replicate early Netflix's protocol.

## 🛠️ System Architecture

This application balances algorithmic complexity with real-time web performance by splitting incoming requests into two distinct processing routes based on data state:

### 1. Predefined Profiles (The SVD Pathway)
For static historical dataset users selected from the sidebar, recommendations are driven by **Truncated Singular Value Decomposition (SVD)**. 

### 2. Live Interactive Profiles (The Fallback Heuristic)
SVD is mathematically rigid and suffers from the **Cold-Start Problem**. When a user checks custom movie boxes in the app, they act as a brand-new entity absent from the trained coordinate space. Retraining a massive SVD matrix synchronously on a button click is completely unfeasible for a web server.

---

## 🚀 What's Next

I am going to start working on a Two-Towers project now to learn more about modern recommendation systems.