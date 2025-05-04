import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# Titre
st.title("🔍 Dashboard Analyse de la résiliation client")

# 1. Chargement des données
df = pd.read_csv("churn_clients.csv")
st.subheader("1. Aperçu des données")
st.write("Nombre de clients :", df.shape[0])
st.write("Colonnes :", list(df.columns))
st.dataframe(df.head())

# 2. Nettoyage
st.subheader("2. Nettoyage des données")
df_clean = df.copy()

# Encodage des variables catégorielles
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

# Gestion des valeurs manquantes
df_clean = df_clean.dropna()
st.write("Données après nettoyage :", df_clean.shape)

# Comptage des valeurs manquantes et des doublons
missing_values = df_clean.isnull().sum().sum()
duplicates = df_clean.duplicated().sum()

# Affichage des résultats
st.write("Nombre total de valeurs manquantes : ", missing_values)
st.write("Nombre de doublons : ", duplicates)

# Normalisation
scaler = StandardScaler()
X = df_clean.drop("Resilie", axis=1)
y = df_clean["Resilie"]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. Visualisations
st.subheader("3. Visualisation des données")
col1, col2 = st.columns(2)
with col1:
    st.write("Histogramme des âges")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.write("Histogramme des revenus")
    fig, ax = plt.subplots()
    sns.histplot(df["Revenu"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

st.write("Corrélation entre satisfaction et résiliation")
fig, ax = plt.subplots()
sns.boxplot(x="Resilie", y="Score_satisfaction", data=df)
st.pyplot(fig)

# 4. Modélisation
st.subheader("4. Entraînement du modèle")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Arbre de décision avec hyperparamètres
params = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='f1')
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Prédiction et évaluation
y_pred = model.predict(X_test)
st.write("Classification Report :")
st.text(classification_report(y_test, y_pred))

st.write("Matrice de confusion")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

# 5. Importance des variables
st.subheader("5. Importance des variables")
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
top_features.plot(kind="bar", ax=ax)
plt.title("Importance des variables")
st.pyplot(fig)

# Suppression des variables peu importantes (importance < 1%)
threshold = 0.01
selected_features = top_features[top_features > threshold].index.tolist()
X_final = X_scaled[selected_features]
st.write(f"✅ Variables conservées après sélection : {selected_features}")

# Nouvelle séparation train/test avec variables sélectionnées
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Optimisation des hyperparamètres avec GridSearchCV
params = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='f1')
grid.fit(X_train_f, y_train_f)
best_model = grid.best_estimator_

# Validation croisée
cv_scores = cross_val_score(best_model, X_final, y, cv=5, scoring='f1')
st.write(f"📊 Score F1 moyen (validation croisée) : {cv_scores.mean():.4f}")
st.write("🔧 Meilleurs hyperparamètres :")
st.json(grid.best_params_)

# Évaluation finale sur test set
y_pred_final = best_model.predict(X_test_f)
st.write("📄 Classification Report :")
st.text(classification_report(y_test_f, y_pred_final))

st.write("📉 Matrice de confusion")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test_f, y_pred_final), annot=True, fmt='d', cmap='BuPu')
st.pyplot(fig)

# Nouvelle importance des variables
st.write("📌 Nouvelle importance des variables après amélioration")
final_importances = pd.Series(best_model.feature_importances_, index=X_final.columns)
sorted_importances = final_importances.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sorted_importances.plot(kind="bar", ax=ax)
plt.title("Importance des variables (modèle optimisé)")
st.pyplot(fig)

# 7. Explication locale avec SHAP
st.subheader("7. Interprétation avec SHAP")
explainer = shap.Explainer(model, X_final)
shap_values = explainer(X_final)

# Sélectionner un index de client
selected_index = st.number_input("Choisir un index client", min_value=0, max_value=len(X_final)-1, step=1)
pred = model.predict([X_final.iloc[selected_index]])
st.write(f"Prédiction pour le client {selected_index} : {'Résilie' if pred[0]==1 else 'Ne résilie pas'}")

# Affichage de l'explication SHAP pour un seul client sélectionné


