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
st.title("🔍 Dashboard Analyse de la Résiliation Client")

# 1. Chargement des données
df = pd.read_csv("churn_clients.csv")
st.subheader("1. Aperçu des données")
st.write("Nombre de clients :", df.shape[0])
st.write("Colonnes :", list(df.columns))
st.dataframe(df.head())

# 2. Nettoyage des données
st.subheader("2. Nettoyage des données")
df_clean = df.copy()

# Encodage des variables catégorielles et nettoyage
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

df_clean = df_clean.dropna()

missing_values = df_clean.isnull().sum().sum()
duplicates = df_clean.duplicated().sum()

st.write("Nombre total de valeurs manquantes :", missing_values)
st.write("Nombre de doublons :", duplicates)

# Standardisation des variables
scaler = StandardScaler()
X = df_clean.drop("Resilie", axis=1)
y = df_clean["Resilie"]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. Visualisations avec un bouton pour les afficher
st.subheader("3. Visualisations des données")

# Boutons pour afficher les graphiques
if st.button("Afficher Histogrammes"):
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

if st.button("Afficher Corrélation"):
    st.write("Corrélation entre satisfaction et résiliation")
    fig, ax = plt.subplots()
    sns.boxplot(x="Resilie", y="Score_satisfaction", data=df.dropna(subset=["Score_satisfaction"]), ax=ax)
    st.pyplot(fig)

# 4. Entraînement du modèle avec un bouton pour afficher les résultats
st.subheader("4. Entraînement du modèle")
if st.button("Entraîner le modèle et afficher les résultats"):
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

    # Validation croisée
    st.subheader("📊 Évaluation du modèle")
    scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
    cv_results = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)

    score_df = pd.DataFrame({
        'Accuracy': [cv_results['test_accuracy'].mean()],
        'Recall': [cv_results['test_recall'].mean()],
        'F1 Score': [cv_results['test_f1'].mean()],
        'AUC': [cv_results['test_roc_auc'].mean()]
    })
    st.dataframe(score_df.T.rename(columns={0: "Score moyen"}).style.format("{:.4f}"))

# 5. Visualisation de l'importance des variables
if st.button("Afficher Importance des Variables"):
    st.subheader("5. Importance des variables")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="bar", ax=ax)
    plt.title("Importance des variables")
    st.pyplot(fig)

# 6. Explication des prédictions avec SHAP
if st.button("Afficher Explication des Prédictions SHAP"):
    st.subheader("6. Explication des prédictions")
    X_final_df = pd.DataFrame(X_scaled, columns=X.columns)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final_df)

    # Sélection de l'observation à analyser
    observation_idx = st.number_input("Choisissez un index client à analyser", min_value=0, max_value=len(X_final_df) - 1, step=1)

    # Prédiction pour cet utilisateur
    prediction = model.predict([X_scaled.iloc[observation_idx]])

    # Affichage de la prédiction
    prediction_label = "❌ Résilie" if prediction[0] == 1 else "✅ Ne résilie pas"
    st.markdown(f"### Pour l'Observation {observation_idx + 1}, le modèle a prédit que le client : **{prediction_label}**")

    # Affichage des valeurs SHAP pour chaque feature
    st.markdown("#### Impact des variables :")
    for feature, shap_value in zip(X_final_df.columns, shap_values[1][observation_idx]):
        direction = "augmente" if shap_value > 0 else "diminue"
        st.markdown(
            f"- **{feature}** : La valeur SHAP est **{shap_value:+.4f}**, ce qui indique que la variable **{feature}** {'augmente' if shap_value > 0 else 'diminue'} la probabilité de résiliation."
        )

    # Calcul de l'impact total
    expected_value = explainer.expected_value[1]
    impact_total = expected_value + shap_values[1][observation_idx].sum()

    # Résumé global
    st.markdown("#### Conclusion :")
    st.markdown(f"La combinaison de ces impacts donne une sortie modèle de **{impact_total:.4f}**.")

