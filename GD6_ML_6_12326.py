import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
import plotly.express as px

# ===========================================================
#                   GLOBAL UI STYLING
# ===========================================================
st.set_page_config(page_title="Premium Clustering App", layout="wide")

st.markdown("""
<style>
    body {
        background: linear-gradient(to bottom right, #0f72aa, #1e293b);
        color: white !important;
    }

    .card {
        padding: 20px;
        background-color: #1e293b;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #3b82f6;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }

    .stButton>button:hover{
        background-color: #2563eb;
    }

    .stTabs[data-baseweb="tab"] {
        background-color: #1e293b !important;
        color: white !important;
        margin-bottom: 8px !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================
#                        HEADER
# ===========================================================
st.markdown("""
<h1 style="text-align:center; color:#fff; font-weight:700;">
    Premium Clustering App
</h1>
<p style="text-align:center; color:#0dc5ed;">
KMeans + DBSCAN + Prediction + Outlier Detection • 2D & 3D Visualization
</p>
""", unsafe_allow_html=True)

# ===========================================================
#                        TAB SETUP
# ===========================================================
tab1, tab2, tab3 = st.tabs([
    "🔵 KMeans - blobs (with Outlier Detection)",
    "🟠 DBSCAN – Moons",
    "ℹ️ Informasi Model & Dataset"
])

# ===========================================================
#            TAB 1 — KMEANS MAKE BLOBS (PREMIUM)
# ===========================================================
with tab1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔵 K-Means Clustering – Make Blobs")

    st.info("""
📌 **Informasi Praktikan**:

**Rentang nilai dataset Make Blobs:**  
- Feature 1: **-5 .. 5**  
- Feature 2: **-5 .. 7**

Namun **rentang fitur bukan batas cluster!**  
Outlier ditentukan oleh **jarak terhadap centroid**.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

    # Custom Outlier Detection Info
    st.markdown("""
**Custom Outlier Detection:**  
- Jika jarak titik baru > (mean jarak cluster + 2×std) = **OUTLIER**

Visualisasi 2D menampilkan *lingkaran radius threshold* untuk memudahkan analisis.
""")

    st.markdown('<div/>', unsafe_allow_html=True)

    # Dataset
    X_blobs, _ = make_blobs(
        n_samples=150,
        centers=3,
        cluster_std=0.85,
        shuffle=True,
        random_state=0
    )

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels_blobs = kmeans.fit_predict(X_blobs)
    centroids = kmeans.cluster_centers_

    col1, col2 = st.columns([1, 2])

    # Predict Panel
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧮 Predict Cluster")

        f1 = st.number_input("Feature 1 (Blobs)", value=0.0, key="b1")
        f2 = st.number_input("Feature 2 (Blobs)", value=0.0, key="b2")

        new_point = np.array([[f1, f2]])
        new_label = None
        is_outlier = False

        if st.button("Predict (KMeans)", key="predict_blobs"):

            new_label = int(kmeans.predict(new_point)[0])

            # Calculate distances
            cluster_pts = X_blobs[labels_blobs == new_label]
            distances = np.linalg.norm(cluster_pts - centroids[new_label], axis=1)

            threshold = distances.mean() + 2 * distances.std()
            new_dist = np.linalg.norm(new_point - centroids[new_label])

            if new_dist > threshold:
                is_outlier = True
                st.error(f"⚠️ OUTLIER – terlalu jauh dari centroid!")
            else:
                st.success(f"✓ Cluster = {new_label}")

            st.info(f"Jarak: {new_dist:.3f} | Threshold: {threshold:.3f}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===================== 2D VISUALIZATION =====================
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 2D Visualization – KMeans")

        fig2d = px.scatter(
            x=X_blobs[:, 0],
            y=X_blobs[:, 1],
            color=labels_blobs.astype(str),
            template="plotly_dark",
            title="KMeans Clustering (Make Blobs)"
        )

        # Centroids
        fig2d.add_scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(size=12, color="white", symbol="x"),
            name="Centroids"
        )

        # Prediction
        if new_label is not None:
            fig2d.add_scatter(
                x=[f1], y=[f2],
                mode="markers",
                marker=dict(size=22, color="red"),
                name="Predicted Point"
            )

        st.plotly_chart(fig2d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===================== 3D VISUALIZATION =====================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 3D Visualization – KMeans")

    fig3d = px.scatter_3d(
        x=X_blobs[:, 0],
        y=X_blobs[:, 1],
        z=np.zeros(len(X_blobs)),
        color=labels_blobs.astype(str),
        template="plotly_dark",
        title="3D KMeans Clustering"
    )

    if new_label is not None:
        fig3d.add_scatter3d(
            x=[f1], y=[f2], z=[0],
            marker=dict(size=12, color="red"),
            name="New Point"
        )

    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================
#                    TAB 2 — DBSCAN MAKE MOONS
# ===========================================================
with tab2:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🟠 DBSCAN Clustering – Make Moons")

    st.info("""
📌 **Informasi Praktikan**:

Rentang dataset Make Moons:
- Feature 1: **-1.5 .. 2.5**
- Feature 2: **-1 .. 1.5**

DBSCAN otomatis memberi label **-1** untuk OUTLIER.
""")

    st.markdown("</div>", unsafe_allow_html=True)

    X_moons, _ = make_moons(n_samples=300, noise=0.07, random_state=0)
    db = DBSCAN(eps=0.28, min_samples=5)
    labels_moons = db.fit_predict(X_moons)

    colA, colB = st.columns([1, 2])

    # Prediction Panel
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧮 Predict (DBSCAN)")

        mx1 = st.number_input("Feature 1 (Moons)", value=0.0, key="m1")
        mx2 = st.number_input("Feature 2 (Moons)", value=0.0, key="m2")

        new_moon = np.array([[mx1, mx2]])
        moon_label = None

        if st.button("Predict (DBSCAN)", key="predict_moon"):
            stacked = np.vstack([X_moons, new_moon])
            moon_label = db.fit_predict(stacked)[-1]

            if moon_label == -1:
                st.error("⚠️ OUTLIER / NOISE")
            else:
                st.success(f"✓ Cluster = {moon_label}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===================== Visualization =====================
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 2D Visualization – DBSCAN")

        fig2_moon = px.scatter(
            x=X_moons[:, 0],
            y=X_moons[:, 1],
            color=labels_moons.astype(str),
            template="plotly_dark",
            title="DBSCAN Clustering (Make Moons)"
        )

        if moon_label is not None:
            fig2_moon.add_scatter(
                x=[mx1], y=[mx2],
                mode='markers',
                marker=dict(size=20, color="red"),
                name="New Point"
            )

        st.plotly_chart(fig2_moon, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===================== 3D Visualization =====================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 3D Visualization – DBSCAN")

    fig3_moon = px.scatter_3d(
        x=X_moons[:, 0],
        y=X_moons[:, 1],
        z=np.zeros(len(X_moons)),
        color=labels_moons.astype(str),
        template="plotly_dark",
        title="3D DBSCAN Clustering"
    )

    if moon_label is not None:
        fig3_moon.add_scatter3d(
            x=[mx1], y=[mx2], z=[0],
            marker=dict(size=12, color="red"),
            name="New Point"
        )

    st.plotly_chart(fig3_moon, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================
#                     TAB 3 — INFORMASI MODEL
# ===========================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Informasi Model, Dataset, dan Pickle")

    st.markdown("""
## 🔍 Mengapa Tidak Menggunakan Pickle?

Dataset sintetis (`make_blobs` dan `make_moons`) **berubah setiap run**, sehingga:
- Model pickle tidak relevan
- Prediksi akan salah

Pickle hanya dipakai jika:
- Dataset tetap & konsisten
- Model dilatih di luar Streamlit menggunakan dataset asli (Kaggle)
    """)

    st.markdown("""
### 🧠 Jika Ingin Menggunakan Model di Dataset Asli:

1. Ambil dataset asli (Kaggle)  
2. Lakukan EDA untuk melihat:  
   - Outlier?  
   - Data bulat / memanjang / melengkung?  
3. Pilih algoritma:
   - KMeans → bentuk bulat / sederhana  
   - DBSCAN → bentuk tidak beraturan + banyak noise  
   - Hierarchical → struktur cluster  
4. Latih model  
5. Simpan pickle  
6. Gunakan pickle di Streamlit
    """)

    st.markdown("""
### 📘 Rule of Thumb

| Pola Data | Algoritma |
|----------|-----------|
| Bulat / rapi | KMeans |
| Bertingkat | Hierarchical |
| Melengkung | DBSCAN |
| Banyak noise | DBSCAN |
| Tidak tahu jumlah cluster | DBSCAN / Hierarchical |

---

### 🎯 Inti Pembelajaran
- Dataset sintetis → latih langsung dalam app  
- Dataset asli → boleh pickle  
- Model terbaik → tergantung bentuk data (*"No Free Lunch"*)
""")

    st.markdown('</div>', unsafe_allow_html=True)
