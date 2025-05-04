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

st.title("🔍 Dashboard Analyse de la résiliation client")

# Chargement des données
df = pd.read_csv("churn_clients.csv")

# 1. Aperçu des données
if st.sidebar.button("1. Aperçu des données"):
    st.subheader("1. Aperçu des données")
    st.write("Nombre de clients :", df.shape[0])
    st.write("Colonnes :", list(df.columns))
    st.dataframe(df.head())

# 2. Nettoyage des données
if st.sidebar.button("2. Nettoyage des données"):
    st.subheader("2. Nettoyage des données")
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

# 3. Visualisations
if st.sidebar.button("3. Visualisation des données"):
    st.subheader("3. Visualisation des données")
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

# 4. Comparaison des modèles
if st.sidebar.button("4. Comparaison des modèles IA"):
    st.subheader("4. Comparaison des modèles IA")

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Modèle de base
    model_base = DecisionTreeClassifier(random_state=42)
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)

    st.write("### 4.1 Résultats du modèle de base")
    st.write("Classification Report :")
    st.text(classification_report(y_test, y_pred_base))

    st.write("Matrice de confusion du modèle de base")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_base), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Modèle amélioré
    model_improved = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    model_improved.fit(X_train, y_train)
    y_pred_improved = model_improved.predict(X_test)

    st.write("### 4.2 Résultats du modèle amélioré")
    st.write("Classification Report :")
    st.text(classification_report(y_test, y_pred_improved))

    st.write("Matrice de confusion du modèle amélioré")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_improved), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Evaluation croisée pour comparer les deux modèles
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

    cv_results_base = cross_validate(model_base, X_scaled, y, cv=cv, scoring=scoring)
    cv_results_improved = cross_validate(model_improved, X_scaled, y, cv=cv, scoring=scoring)

    st.write("### 4.3 Scores Validation croisée")

    # Scores validation croisée du modèle de base
    score_df_base = pd.DataFrame({
        'Accuracy': [cv_results_base['test_accuracy'].mean()],
        'Recall': [cv_results_base['test_recall'].mean()],
        'F1 Score': [cv_results_base['test_f1'].mean()],
        'AUC': [cv_results_base['test_roc_auc'].mean()]
    })

    st.write("Modèle de base - Scores Validation croisée :")
    st.dataframe(score_df_base.T.rename(columns={0: "Score moyen"}).style.format("{:.4f}"))

    # Scores validation croisée du modèle amélioré
    score_df_improved = pd.DataFrame({
        'Accuracy': [cv_results_improved['test_accuracy'].mean()],
        'Recall': [cv_results_improved['test_recall'].mean()],
        'F1 Score': [cv_results_improved['test_f1'].mean()],
        'AUC': [cv_results_improved['test_roc_auc'].mean()]
    })

    st.write("Modèle amélioré - Scores Validation croisée :")
    st.dataframe(score_df_improved.T.rename(columns={0: "Score moyen"}).style.format("{:.4f}"))

    # Importance des variables - Modèle amélioré
    st.write("### 4.4 Importance des variables du modèle amélioré")
    importances = pd.Series(model_improved.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="bar", ax=ax)
    plt.title("Importance des variables (modèle amélioré)")
    st.pyplot(fig)
