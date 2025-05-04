import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

# ----------------- Chargement et Prétraitement des données -----------------
@st.cache_data
def load_and_preprocess_data():
    # Remplace le chemin par le fichier CSV contenant tes données
    data = pd.read_csv("churn_clients.csv")  # Assure-toi que le fichier est dans le bon répertoire
    return data

data = load_and_preprocess_data()

# ----------------- Prétraitement pour le modèle -----------------
st.sidebar.title("Filtres")
anciennete = st.sidebar.slider("Ancienneté (mois)", int(data['Anciennete'].min()), int(data['Anciennete'].max()))
frequence = st.sidebar.slider("Fréquence d'utilisation", float(data['Frequence_utilisation'].min()), float(data['Frequence_utilisation'].max()))
score = st.sidebar.slider("Score de satisfaction", float(data['Score_satisfaction'].min()), float(data['Score_satisfaction'].max()))

filtered_data = data[
    (data['Anciennete'] >= anciennete) &
    (data['Frequence_utilisation'] >= frequence) &
    (data['Score_satisfaction'] >= score)
]

# ----------------- Entraînement du modèle KMeans -----------------
X = filtered_data[['Anciennete', 'Frequence_utilisation', 'Score_satisfaction']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
filtered_data['Cluster'] = kmeans.fit_predict(X_scaled)

# ----------------- Entraînement du modèle pour SHAP -----------------
# Pour l'explication SHAP, utilisons un arbre de décision
y = [1 if x < 3 else 0 for x in filtered_data['Score_satisfaction']]  # Simplification : résiliation si Score < 3

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_scaled, y)

# Calcul de l'explainer SHAP
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_scaled)

# ----------------- Dashboard -----------------
st.title("📊 Dashboard Clients - Résiliations et Clustering")

st.markdown("Analyse interactive des comportements clients pour identifier les motifs de résiliation.")

st.subheader("📌 Données filtrées")
st.write(f"Nombre de clients : {filtered_data.shape[0]}")
st.dataframe(filtered_data.head())

# ----------------- Clustering -----------------
st.subheader("🧠 Clustering des clients (KMeans)")

fig_cluster = px.scatter_3d(filtered_data, 
                            x='Anciennete', y='Frequence_utilisation', z='Score_satisfaction',
                            color='Cluster', title="Clustering 3D des clients")
st.plotly_chart(fig_cluster)

# ----------------- SHAP -----------------
st.subheader("📈 Explication des Résiliations avec SHAP")

client_id = st.selectbox("Choisir un client à expliquer :", filtered_data.index)
client_data = filtered_data.loc[[client_id]][['Anciennete', 'Frequence_utilisation', 'Score_satisfaction']]

shap_value_client = shap_values[1][client_id]  # 1 pour la classe 'résilié' (si Score < 3)

st.markdown("### Contribution des variables")
fig_shap = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_value_client, client_data.iloc[0], show=False)
st.pyplot(fig_shap)

# ----------------- Résumé -----------------
st.subheader("📌 Résumé du cluster sélectionné")
selected_cluster = st.selectbox("Choisir un cluster :", sorted(filtered_data['Cluster'].unique()))
cluster_data = filtered_data[filtered_data['Cluster'] == selected_cluster]

st.metric("Taille du cluster", len(cluster_data))
st.metric("Satisfaction moyenne", round(cluster_data['Score_satisfaction'].mean(), 2))
st.metric("Ancienneté moyenne", round(cluster_data['Anciennete'].mean(), 2))

fig_hist = px.histogram(cluster_data, x='Score_satisfaction', nbins=20, title="Distribution de satisfaction dans le cluster")
st.plotly_chart(fig_hist)
