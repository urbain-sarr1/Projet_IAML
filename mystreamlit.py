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

# Le reste du code pour l'entraînement et les autres sections n'est pas visible en fonction des boutons
