import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations

st.set_page_config(page_title="Analisis Resep Obat", layout="wide")
st.title("🧪 Analisis Pola Resep Obat Diabetes Melitus Tipe 2")

uploaded_file = st.file_uploader("📂 Unggah file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("🔍 Preview Data Mentah")
    st.dataframe(df.head())

    tab1, tab2 = st.tabs(["📊 Demografi → Obat", "💊 Kombinasi Obat"])

    # =======================
    # TAB 1 – Segmentasi Pasien (Demografi → Obat)
    # =======================
    with tab1:
        st.sidebar.header("🔎 Filter Segmentasi Pasien")
        jk_filter = st.sidebar.selectbox("Jenis Kelamin", ["Semua", "Laki - Laki", "Perempuan"])
        usia_filter = st.sidebar.selectbox("Kelompok Usia", ["Semua", "Under 30", "30-39", "40-49", "50-59", "60-69", "70 and above"])
        tekanan_filter = st.sidebar.selectbox("Tekanan Darah", ["Semua", "Normal BP", "Elevated BP", "High BP"])

        filtered_df = df.copy()
        if jk_filter != "Semua":
            filtered_df = filtered_df[filtered_df["Jenis Kelamin"] == jk_filter]
        if usia_filter != "Semua":
            filtered_df = filtered_df[filtered_df["Usia"] == usia_filter]
        if tekanan_filter != "Semua":
            filtered_df = filtered_df[filtered_df["Tekanan Darah"] == tekanan_filter]

        transaksi = filtered_df.apply(
            lambda row: [
                str(row['Jenis Kelamin']).strip(),
                str(row['Usia']).strip(),
                str(row['Tekanan Darah']).strip()
            ] + [item.strip() for item in str(row['Resep Obat']).split(',') if item.strip()],
            axis=1
        ).tolist()

        te = TransactionEncoder()
        df_encoded = pd.DataFrame(te.fit(transaksi).transform(transaksi), columns=te.columns_)

        st.subheader("🧾 Data Transaksi (Encoded)")
        st.dataframe(df_encoded.head())

        min_support = st.slider("Minimum Support", 0.005, 1.0, 0.03, 0.005)
        min_confidence = st.slider("Minimum Confidence", 0.05, 1.0, 0.6, 0.05)
        use_lift = st.checkbox("Gunakan Filter Lift", value=True)
        if use_lift:
            min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.5, 0.1)

        if st.button("🚀 Jalankan Apriori Segmentasi"):
            itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
            kategori_demografi = set(
                ["Laki - Laki", "Perempuan", "Under 30", "30-39", "40-49", "50-59", "60-69", "70 and above",
                 "Normal BP", "Elevated BP", "High BP"])

            rules = rules[
                rules["antecedents"].apply(lambda x: all(i in kategori_demografi for i in x)) &
                rules["consequents"].apply(lambda x: all(i not in kategori_demografi for i in x))
                ]
            if use_lift:
                rules = rules[rules['lift'] > min_lift]

            if rules.empty:
                st.warning("⚠ Tidak ditemukan aturan.")
            else:
                st.success(f"✅ {len(rules)} aturan ditemukan.")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                # Scatter Plot
                st.subheader("📈 Scatter Plot: Lift vs Confidence")
                fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=rules,
                    x="confidence",
                    y="lift",
                    size="support",
                    sizes=(100, 500),
                    legend="brief",
                    ax=ax_scatter1
                )
                for i, row in rules.iterrows():
                    ax_scatter1.text(row['confidence'], row['lift'], f"{row['support']:.3f}", fontsize=9, ha='center')
                ax_scatter1.set_title("Scatter Plot: Confidence vs Lift (Segmentasi Demografi)")
                ax_scatter1.grid(True)
                st.pyplot(fig_scatter1)

                # Heatmap Kombinasi Obat berdasarkan segmentasi
                st.subheader("🧪 Heatmap Kombinasi Obat (Hasil Filter Segmentasi)")
                resep_list_seg = filtered_df['Resep Obat'].dropna().apply(
                    lambda x: [item.strip() for item in str(x).split(',') if item.strip()]
                ).tolist()
                all_items_seg = pd.Series([item for sublist in resep_list_seg for item in sublist]).unique().tolist()
                co_matrix_seg = pd.DataFrame(0, index=all_items_seg, columns=all_items_seg)
                for resep in resep_list_seg:
                    for item_a, item_b in combinations(resep, 2):
                        co_matrix_seg.loc[item_a, item_b] += 1
                        co_matrix_seg.loc[item_b, item_a] += 1
                    for item in resep:
                        co_matrix_seg.loc[item, item] += 1

                fig_heat1, ax_heat1 = plt.subplots(figsize=(14, 10))
                sns.heatmap(co_matrix_seg, cmap="YlGnBu", annot=False, fmt="d", linewidths=0.5, ax=ax_heat1)
                ax_heat1.set_title("Heatmap Frekuensi Kombinasi Obat (Segmentasi)")
                st.pyplot(fig_heat1)

                # Network Graph
                # Network Graph dengan label edge dan bobot dinamis
                st.subheader("🔗 Network Graph: Relasi Aturan Asosiasi (Warna Berdasarkan Confidence)")
                G = nx.DiGraph()
                for _, row in rules.iterrows():
                    for a in row['antecedents']:
                        for c in row['consequents']:
                            G.add_edge(a, c, weight=row['lift'], confidence=row['confidence'],
                                       label=f"{row['confidence']:.2f}")

                fig_net, ax_net = plt.subplots(figsize=(12, 8))
                pos = nx.spring_layout(G, k=0.7, seed=42)

                # Node
                nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', alpha=0.7, ax=ax_net)
                nx.draw_networkx_labels(G, pos, font_size=9, font_color='black', ax=ax_net)

                # Ambil nilai confidence untuk edge coloring
                edges = G.edges(data=True)
                edge_conf = [d['confidence'] for (_, _, d) in edges]

                # Normalize warna
                norm = plt.Normalize(min(edge_conf), max(edge_conf))
                cmap = plt.cm.viridis  # Bisa diganti dengan cmap lain seperti plt.cm.coolwarm, plt.cm.plasma

                # Gambar edge dengan warna berdasarkan confidence
                edges_drawn = nx.draw_networkx_edges(
                    G, pos, edgelist=edges,
                    edge_color=edge_conf,
                    edge_cmap=cmap,
                    edge_vmin=min(edge_conf),
                    edge_vmax=max(edge_conf),
                    width=2,
                    arrows=True,
                    ax=ax_net
                )

                # Tambahkan label confidence
                edge_labels = {(u, v): f"{d['confidence']:.2f}" for u, v, d in edges}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax_net)

                # Tambahkan colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_net, orientation="vertical")
                cbar.set_label("Confidence")

                ax_net.set_title("Network Graph Aturan Asosiasi", fontsize=12)
                ax_net.axis('off')
                st.pyplot(fig_net)

                # Frequent Itemset ≥ 2 items
                st.subheader("📋 Frequent Itemset (≥ 2 Item)")
                frequent_2plus = itemsets[itemsets['itemsets'].apply(lambda x: len(x) >= 2)]
                st.dataframe(frequent_2plus)

    # =======================
    # TAB 2 – Kombinasi Obat Tanpa Segmentasi
    # =======================
    with tab2:
        resep_list = df['Resep Obat'].dropna().apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()]
        ).tolist()

        te2 = TransactionEncoder()
        df_encoded2 = pd.DataFrame(te2.fit(resep_list).transform(resep_list), columns=te2.columns_)

        st.subheader("🧾 Data Transaksi Obat (Encoded)")
        st.dataframe(df_encoded2.head())

        min_support2 = st.slider("Minimum Support", 0.005, 1.0, 0.03, 0.005, key="sup2")
        min_confidence2 = st.slider("Minimum Confidence", 0.05, 1.0, 0.35, 0.05, key="conf2")
        use_lift2 = st.checkbox("Gunakan Filter Lift", value=True, key="lift2")
        if use_lift2:
            min_lift2 = st.slider("Minimum Lift", 1.0, 5.0, 1.5, 0.1, key="liftval2")

        if st.button("🚀 Jalankan Apriori Kombinasi Obat"):
            itemsets2 = apriori(df_encoded2, min_support=min_support2, use_colnames=True)
            rules2 = association_rules(itemsets2, metric="confidence", min_threshold=min_confidence2)

            kategori_demografi = set(["Laki - Laki", "Perempuan", "Under 30", "30-39", "40-49", "50-59", "60-69", "70 and above", "Normal BP", "Elevated BP", "High BP"])
            rules2 = rules2[
                rules2["antecedents"].apply(lambda x: all(i not in kategori_demografi for i in x)) &
                rules2["consequents"].apply(lambda x: all(i not in kategori_demografi for i in x))
            ]
            if use_lift2:
                rules2 = rules2[rules2['lift'] > min_lift2]

            if rules2.empty:
                st.warning("⚠ Tidak ditemukan aturan asosiasi.")
            else:
                st.success(f"✅ {len(rules2)} aturan ditemukan.")
                st.dataframe(rules2[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

                # Scatter Plot
                st.subheader("📈 Scatter Plot: Lift vs Confidence")
                fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=rules2,
                    x="confidence",
                    y="lift",
                    size="support",
                    sizes=(100, 500),
                    legend="brief",
                    ax=ax_scatter2
                )
                for i, row in rules2.iterrows():
                    ax_scatter2.text(row['confidence'], row['lift'], f"{row['support']:.3f}", fontsize=9, ha='center')
                ax_scatter2.set_title("Scatter Plot: Confidence vs Lift (Kombinasi Obat)")
                ax_scatter2.grid(True)
                st.pyplot(fig_scatter2)

                # Heatmap
                st.subheader("🧪 Heatmap Kombinasi Obat")
                all_items = pd.Series([item for sublist in resep_list for item in sublist]).unique().tolist()
                co_matrix = pd.DataFrame(0, index=all_items, columns=all_items)
                for resep in resep_list:
                    for item_a, item_b in combinations(resep, 2):
                        co_matrix.loc[item_a, item_b] += 1
                        co_matrix.loc[item_b, item_a] += 1
                    for item in resep:
                        co_matrix.loc[item, item] += 1

                fig_heat2, ax_heat2 = plt.subplots(figsize=(14, 10))
                sns.heatmap(co_matrix, cmap="YlGnBu", annot=False, fmt="d", linewidths=0.5, ax=ax_heat2)
                ax_heat2.set_title("Heatmap Frekuensi Kombinasi Obat")
                st.pyplot(fig_heat2)

                # Network Graph
                st.subheader("🔗 Network Graph: Relasi Kombinasi Obat")

                G2 = nx.DiGraph()
                for _, row in rules2.iterrows():
                    for a in row['antecedents']:
                        for c in row['consequents']:
                            G2.add_edge(a, c,
                                        weight=row['lift'],
                                        confidence=row['confidence'],
                                        label=f"{row['confidence']:.2f}")

                fig_net2, ax_net2 = plt.subplots(figsize=(12, 8))

                # 🔧 K = Jarak antar node → Naikkan agar tidak menumpuk
                pos2 = nx.spring_layout(G2, k=2.5, iterations=100, seed=42)

                # 🎯 Gambar node
                nx.draw_networkx_nodes(
                    G2, pos2,
                    node_size=1800,  # Ukuran node lebih kecil dari default
                    node_color='skyblue',
                    alpha=0.8,
                    ax=ax_net2
                )

                # 🏷️ Gambar label node
                nx.draw_networkx_labels(
                    G2, pos2,
                    font_size=9,
                    font_color='black',
                    ax=ax_net2
                )

                # 🖌️ Edge dengan warna berdasarkan confidence
                edges2 = G2.edges(data=True)
                edge_conf2 = [d['confidence'] for (_, _, d) in edges2]

                # 🎨 Warna edge berdasarkan confidence
                norm2 = plt.Normalize(min(edge_conf2), max(edge_conf2))
                cmap2 = plt.cm.viridis

                nx.draw_networkx_edges(
                    G2, pos2,
                    edgelist=edges2,
                    edge_color=edge_conf2,
                    edge_cmap=cmap2,
                    edge_vmin=min(edge_conf2),
                    edge_vmax=max(edge_conf2),
                    width=2,
                    arrows=True,
                    ax=ax_net2
                )

                # ✏️ Label confidence pada garis
                edge_labels2 = {(u, v): f"{d['confidence']:.2f}" for u, v, d in edges2}
                nx.draw_networkx_edge_labels(
                    G2, pos2,
                    edge_labels=edge_labels2,
                    font_size=8,
                    ax=ax_net2
                )

                # 🎚️ Tambahkan colorbar
                sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
                sm2.set_array([])
                cbar2 = plt.colorbar(sm2, ax=ax_net2, orientation="vertical")
                cbar2.set_label("Confidence")

                # 🎯 Finalisasi
                ax_net2.set_title("Network Graph Kombinasi Obat (Warna = Confidence)", fontsize=12)
                ax_net2.axis('off')
                st.pyplot(fig_net2)

                # Frequent Itemset ≥ 2 items
                st.subheader("📋 Frequent Itemset (≥ 2 Item)")
                frequent_2plus2 = itemsets2[itemsets2['itemsets'].apply(lambda x: len(x) >= 2)]
                st.dataframe(frequent_2plus2)