import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap

st.set_page_config(page_title="Analyse Résiliation Client", layout="wide")

# 1. Import des données
st.title("📊 Tableau de bord : Analyse de la Résiliation Client")
uploaded_file = st.file_uploader("📂 Importer votre fichier CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.write(df.head())

    st.markdown(f"**Nombre de clients :** {df.shape[0]}")
    st.markdown(f"**Colonnes :** {list(df.columns)}")

    # Nettoyage
    df = df.dropna()

    # Encodage
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Visualisations
    st.subheader("Visualisation des données")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Histogramme des âges**")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.markdown("**Histogramme des revenus**")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Revenu'], kde=True, ax=ax2)
        st.pyplot(fig2)

    st.markdown("**Corrélation entre satisfaction et résiliation**")
    fig_corr, ax_corr = plt.subplots()
    sns.boxplot(x='Resilie', y='Score_satisfaction', data=df, ax=ax_corr)
    st.pyplot(fig_corr)

    # Séparation
    X = df.drop('Resilie', axis=1)
    y = df['Resilie']

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement avec hyperparamètres et validation croisée
    st.subheader("🔍 Entraînement du modèle")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    params = {'max_depth': [3, 5, 10, 15], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring='f1')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Rapport de performance
    st.text("Rapport de classification :")
    st.text(classification_report(y_test, y_pred))

    st.markdown("**Matrice de confusion :**")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

    # Importance des variables
    st.subheader("🔎 Importance des variables")
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False)
    st.bar_chart(feat_imp)

    # SHAP
    st.subheader("🔍 Interprétation locale avec SHAP")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_scaled)

    idx = st.number_input("Choisis un ID client (entre 0 et {})".format(len(df)-1), min_value=0, max_value=len(df)-1, step=1)

    st.markdown("**Valeurs d'entrée du client :**")
    st.write(df.iloc[idx])
    st.markdown("**Prédiction :**")
    pred = best_model.predict([X_scaled[idx]])[0]
    st.success("Résilie" if pred == 1 else "Ne résilie pas")

    st.markdown("**Explication SHAP pour ce client :**")
    shap.initjs()
    st_shap = st.pyplot()
    shap.plots.waterfall(shap.Explanation(values=shap_values[1][idx], 
                                          base_values=explainer.expected_value[1],
                                          data=X.iloc[idx], 
                                          feature_names=X.columns.tolist()))
