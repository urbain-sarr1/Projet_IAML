import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# Configuration du titre du dashboard
st.set_page_config(page_title="Analyse de la Résiliation Client", page_icon="🔍", layout="wide")

# Titre
st.title("🔍 Dashboard Interactif - Analyse de la Résiliation Client")

# 1. Chargement des données
@st.cache
def load_data():
    return pd.read_csv("churn_clients.csv")

df = load_data()

# 2. Aperçu des données
st.subheader("1. Aperçu des données")
st.write(f"Nombre total de clients : {df.shape[0]}")
st.write(f"Colonnes disponibles : {list(df.columns)}")

# Affichage dynamique de l'aperçu des données
num_rows = st.slider("Sélectionnez le nombre de lignes à afficher", 5, 50, 10)
st.dataframe(df.head(num_rows))

# 3. Sélection des caractéristiques
st.subheader("2. Sélectionnez les caractéristiques à analyser")

# Choix des variables à afficher dans la visualisation
col1, col2 = st.columns(2)
with col1:
    selected_feature = st.selectbox("Sélectionner une caractéristique pour la visualisation", df.columns.tolist())

# Visualisation des caractéristiques sélectionnées
st.write(f"**Histogramme de {selected_feature}**")
fig, ax = plt.subplots()
sns.histplot(df[selected_feature].dropna(), kde=True, ax=ax)
st.pyplot(fig)

# 4. Nettoyage et préparation des données
st.subheader("3. Nettoyage et préparation des données")

df_clean = df.copy()

# Encodage des variables catégorielles
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

# Suppression des valeurs manquantes
df_clean = df_clean.dropna()

# Mise à l'échelle des données
scaler = StandardScaler()
X = df_clean.drop("Resilie", axis=1)
y = df_clean["Resilie"]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 5. Sélection d'un modèle pour prédiction
st.subheader("4. Modélisation et Prédiction")

# Sélection de l'algorithme de prédiction
model_option = st.selectbox("Choisissez le modèle à utiliser", ["Arbre de Décision", "Régression Logistique", "Forêt Aléatoire"])

if model_option == "Arbre de Décision":
    model = DecisionTreeClassifier(random_state=42)
elif model_option == "Régression Logistique":
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
else:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)

# Séparation des données et entraînement
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Affichage des résultats de prédiction
st.write("**Rapport de Classification**")
st.text(classification_report(y_test, y_pred))

# 6. Visualisation de la matrice de confusion
st.subheader("5. Matrice de confusion")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# 7. Explication avec SHAP
st.subheader("6. Explication des Prédictions avec SHAP")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Choisir un client pour expliquer la prédiction
index = st.slider("Choisissez un client à analyser", 0, len(X_scaled) - 1, 0)
shap.initjs()

# Affichage de l'importance des variables pour ce client
st.write(f"Prédiction pour le client {index}: {'Résilie' if y_pred[index] == 1 else 'Ne résilie pas'}")
shap.force_plot(explainer.expected_value[1], shap_values[1][index], X_scaled.iloc[index])

# 8. Visualisation de l'importance des variables
st.subheader("7. Importance des Variables")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
importances.plot(kind="bar", ax=ax)
plt.title("Importance des Variables")
st.pyplot(fig)

# 9. Visualisation interactive
st.subheader("8. Visualisation Interactive avec des Graphiques Dynamiques")

# Sélection dynamique des variables
var1 = st.selectbox("Sélectionnez une variable pour analyser la relation avec la résiliation", df.columns)
var2 = st.selectbox("Sélectionnez une autre variable pour analyser la relation", df.columns)

# Visualisation interactive de la relation entre deux variables
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df[var1], y=df[var2], hue=df["Resilie"], ax=ax, palette="coolwarm")
st.pyplot(fig)

# Conclusion
st.subheader("9. Conclusion")
st.write("Ce tableau de bord permet une analyse approfondie de la résiliation client. Vous pouvez interagir avec les données pour mieux comprendre les facteurs influençant la décision de résiliation.")

