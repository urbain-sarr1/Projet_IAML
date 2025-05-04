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

# Titre du tableau de bord
st.title("🔍 Dashboard Analyse de la résiliation client")

# Menu de navigation (barre latérale)
st.sidebar.title("🧭 Naviguer")

sections = [
    "1. Aperçu des données",
    "2. Nettoyage des données",
    "3. Visualisation des données",
    "4. Entraînement du modèle",
    "5. Importance des variables",
    "6. Amélioration du modèle",
    "7. Explication des prédictions"
]

for section in sections:
    anchor = section.lower().replace(".", "").replace(" ", "-")
    st.sidebar.markdown(f"- [{section}](#{anchor})")

# Ajout des liens pour les ancres de chaque section
sections = [
    "Aperçu des données",
    "Nettoyage des données",
    "Visualisation des données",
    "Entraînement du modèle",
    "Importance des variables",
    "Amélioration du modèle",
    "Explication des prédictions"
]

# Affichage du menu de navigation
for section in sections:
    st.sidebar.markdown(f"[{section}](#{section.lower().replace(' ', '-')})")

# Chargement des données
df = pd.read_csv("churn_clients.csv")

# Tout afficher, section par section
st.markdown("<a id='aperçu-des-données'></a>", unsafe_allow_html=True)
st.markdown("## 1. Aperçu des données")
st.write("Nombre de clients :", df.shape[0])
st.write("Colonnes :", list(df.columns))
num_lines = st.slider("Choisissez le nombre de lignes à afficher", min_value=5, max_value=df.shape[0], step=5, value=10)
st.dataframe(df.head(num_lines))

st.markdown("<a id='nettoyage-des-données'></a>", unsafe_allow_html=True)
st.markdown("## 2. Nettoyage des données")
df_clean = df.copy()

for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

df_clean = df_clean.dropna()

missing_values = df_clean.isnull().sum().sum()
duplicates = df_clean.duplicated().sum()

st.write("Nombre total de valeurs manquantes :", missing_values)
st.write("Nombre de doublons :", duplicates)

scaler = StandardScaler()
X = df_clean.drop("Resilie", axis=1)
y = df_clean["Resilie"]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

st.markdown("<a id='visualisation-des-données'></a>", unsafe_allow_html=True)
st.markdown("## 3. Visualisation des données")
col1, col2 = st.columns(2)
with col1:
    st.write("Histogramme des âges")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"].dropna(), bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Histogramme des revenus")
    fig, ax = plt.subplots()
    sns.histplot(df["Revenu"].dropna(), bins=20, kde=True, ax=ax)
    st.pyplot(fig)

st.write("Corrélation entre satisfaction et résiliation")
fig, ax = plt.subplots()
sns.boxplot(x="Resilie", y="Score_satisfaction", data=df.dropna(subset=["Score_satisfaction"]), ax=ax)
st.pyplot(fig)

st.markdown("<a id='entrainement-du-modèle'></a>", unsafe_allow_html=True)
st.markdown("## 4. Entraînement du modèle")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("Classification Report :")
st.text(classification_report(y_test, y_pred))

st.write("Matrice de confusion")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.markdown("<a id='importance-des-variables'></a>", unsafe_allow_html=True)
st.markdown("## 5. Importance des variables")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="bar", ax=ax)
plt.title("Importance des variables")
st.pyplot(fig)

st.markdown("<a id='amelioration-du-modèle'></a>", unsafe_allow_html=True)
st.markdown("## 6. Amélioration du modèle")
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

best_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
best_model.fit(X_train_f, y_train_f)

cv_final = cross_validate(best_model, X_scaled, y, cv=cv, scoring=['accuracy', 'recall', 'f1', 'roc_auc'])

st.write("📊 Scores validation croisée :")
cv_score_df = pd.DataFrame({
    'Accuracy': [cv_final['test_accuracy'].mean()],
    'Recall': [cv_final['test_recall'].mean()],
    'F1 Score': [cv_final['test_f1'].mean()],
    'AUC': [cv_final['test_roc_auc'].mean()]
})
st.dataframe(cv_score_df.T.rename(columns={0: "Score moyen"}).style.format("{:.4f}"))

y_pred_final = best_model.predict(X_test_f)
st.write("📄 Rapport de classification final :")
st.text(classification_report(y_test_f, y_pred_final))

st.write("📉 Matrice de confusion")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test_f, y_pred_final), annot=True, fmt='d', cmap='BuPu', ax=ax)
st.pyplot(fig)

st.markdown("<a id='explication-des-prédictions'></a>", unsafe_allow_html=True)
st.markdown("## 7. Explication des prédictions")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_scaled)

observation_idx = st.number_input(
    "Choisissez un index client à analyser", 
    min_value=0, 
    max_value=len(X_scaled) - 1, 
    step=1
)

prediction = best_model.predict([X_scaled.iloc[observation_idx]])

prediction_label = "❌ Résilie" if prediction[0] == 1 else "✅ Ne résilie pas"
st.markdown(f"### Pour l'Observation {observation_idx + 1}, le modèle a prédit que le client : **{prediction_label}**")

st.markdown("#### Impact des variables :")
for feature, shap_value in zip(X_scaled.columns, shap_values[1][observation_idx]):
    direction = "augmente" if shap_value > 0 else "diminue"
    st.markdown(
        f"- **{feature}** : La valeur SHAP est **{shap_value:+.4f}**, ce qui indique que la variable **{feature}** {'augmente' if shap_value > 0 else 'diminue'} la probabilité de résiliation."
    )
