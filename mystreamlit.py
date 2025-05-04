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

st.set_page_config(page_title="Dashboard Résiliation Client", layout="wide")

# --- MENU DE NAVIGATION ---
sections = {
    "1. Aperçu des données": "apercu-des-donnees",
    "2. Nettoyage des données": "nettoyage-des-donnees",
    "3. Visualisation des données": "visualisation-des-donnees",
    "4. Entraînement du modèle": "entrainement-du-modele",
    "5. Importance des variables": "importance-des-variables",
    "6. Amélioration du modèle": "amelioration-du-modele",
    "7. Explication des prédictions": "explication-des-predictions"
}

selected_section = st.sidebar.radio("🧭 Navigation", list(sections.keys()))

# Afficher lien vers l'ancre (pour déclencher le scroll au clic)
st.sidebar.markdown(f"[Aller à : {selected_section}](#{sections[selected_section]})")

# --- Titre principal ---
st.title("🔍 Dashboard Analyse de la résiliation client")

# 1. Aperçu des données
st.markdown(f'<h2 id="{sections["1. Aperçu des données"]}">1. Aperçu des données</h2>', unsafe_allow_html=True)
df = pd.read_csv("churn_clients.csv")
st.write("Nombre de clients :", df.shape[0])
st.write("Colonnes :", list(df.columns))
num_lines = st.slider("Choisissez le nombre de lignes à afficher", min_value=5, max_value=df.shape[0], step=5, value=10)
st.dataframe(df.head(num_lines))

# 2. Nettoyage des données
st.markdown(f'<h2 id="{sections["2. Nettoyage des données"]}">2. Nettoyage des données</h2>', unsafe_allow_html=True)
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

# 3. Visualisation des données
st.markdown(f'<h2 id="{sections["3. Visualisation des données"]}">3. Visualisation des données</h2>', unsafe_allow_html=True)
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

# 4. Entraînement du modèle
st.markdown(f'<h2 id="{sections["4. Entraînement du modèle"]}">4. Entraînement du modèle</h2>', unsafe_allow_html=True)
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

# Évaluation croisée
st.markdown("📊 Évaluation du modèle")
scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
cv_results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)
score_df = pd.DataFrame({
    'Accuracy': [cv_results['test_accuracy'].mean()],
    'Recall': [cv_results['test_recall'].mean()],
    'F1 Score': [cv_results['test_f1'].mean()],
    'AUC': [cv_results['test_roc_auc'].mean()]
})
st.dataframe(score_df.T.rename(columns={0: "Score moyen"}).style.format("{:.4f}"))

# 5. Importance des variables
st.markdown(f'<h2 id="{sections["5. Importance des variables"]}">5. Importance des variables</h2>', unsafe_allow_html=True)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="bar", ax=ax)
plt.title("Importance des variables")
st.pyplot(fig)
thresh = 0.01
selected_features = importances[importances > thresh].index.tolist()
X_final = X_scaled[selected_features]
st.write(f"✅ Variables sélectionnées : {selected_features}")

# 6. Amélioration du modèle
st.markdown(f'<h2 id="{sections["6. Amélioration du modèle"]}">6. Amélioration du modèle</h2>', unsafe_allow_html=True)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
best_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
best_model.fit(X_train_f, y_train_f)
cv_final = cross_validate(best_model, X_final, y, cv=cv, scoring=scoring)
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
st.write("📌 Nouvelle importance des variables")
final_importances = pd.Series(best_model.feature_importances_, index=X_final.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
final_importances.plot(kind="bar", ax=ax)
plt.title("Importance des variables (modèle optimisé)")
st.pyplot(fig)

# 7. Explication des prédictions
st.markdown(f'<h2 id="{sections["7. Explication des prédictions"]}">7. Explication des prédictions</h2>', unsafe_allow_html=True)
X_final_df = pd.DataFrame(X_final, columns=X_final.columns)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_final_df)
observation_idx = st.number_input(
    "Choisissez un index client à analyser", 
    min_value=0, 
    max_value=len(X_final_df) - 1, 
    step=1
)
prediction = best_model.predict([X_final.iloc[observation_idx]])
prediction_label = "❌ Résilie" if prediction[0] == 1 else "✅ Ne résilie pas"
st.markdown(f"### Pour l'Observation {observation_idx + 1}, le modèle a prédit que le client : **{prediction_label}**")
st.markdown("#### Impact des variables :")
for feature, shap_value in zip(X_final.columns, shap_values[1][observation_idx]):
    direction = "augmente" if shap_value > 0 else "diminue"
    st.markdown(
        f"- **{feature}** : SHAP = **{shap_value:+.4f}**, ce qui **{direction}** la probabilité de résiliation."
    )
expected_value = explainer.expected_value[1]
impact_total = expected_value + shap_values[1][observation_idx].sum()
st.markdown("#### Conclusion :")
st.markdown(f"La sortie modèle (valeur combinée) est de **{impact_total:.4f}**.")

