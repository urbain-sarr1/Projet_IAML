import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap

# Titre principal
st.title("🔍 Dashboard Analyse de la Résiliation Client")

# 1. Barre de navigation
page = st.sidebar.radio("Sélectionner la page", 
                        ("Aperçu des données", "Nettoyage des données", "Visualisations", 
                         "Entraînement du modèle", "Explication des prédictions"))

# Chargement des données
df = pd.read_csv("churn_clients.csv")

# 2. Aperçu des données
if page == "Aperçu des données":
    st.title("🔍 Aperçu des Données")
    st.write("Nombre de clients :", df.shape[0])
    st.write("Colonnes :", list(df.columns))
    st.dataframe(df.head())

# 3. Nettoyage des données
elif page == "Nettoyage des données":
    st.title("🧹 Nettoyage des données")
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

# 4. Visualisations Interactives
elif page == "Visualisations":
    st.title("📊 Visualisation des Données")

    graph_type = st.radio("Sélectionner un graphique", 
                          ("Histogramme des âges", "Histogramme des revenus", "Corrélation Satisfaction-Resiliation"))

    if graph_type == "Histogramme des âges":
        fig = px.histogram(df, x="Age", nbins=20, title="Distribution des âges des clients")
        st.plotly_chart(fig)

    elif graph_type == "Histogramme des revenus":
        fig = px.histogram(df, x="Revenu", nbins=20, title="Distribution des revenus des clients")
        st.plotly_chart(fig)

    elif graph_type == "Corrélation Satisfaction-Resiliation":
        fig = px.box(df, x="Resilie", y="Score_satisfaction", title="Satisfaction des clients vs Résiliation")
        st.plotly_chart(fig)

# 5. Entraînement du Modèle
elif page == "Entraînement du modèle":
    st.title("⚙️ Entraînement du Modèle")
    
    # Vérification que X_scaled est défini correctement
    if 'X_scaled' not in locals():
        st.error("Les données n'ont pas été préalablement traitées et normalisées.")
    else:
        # Séparation des données
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
        scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
        cv_results = cross_validate(model, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring=scoring)

        score_df = pd.DataFrame({
            'Accuracy': [cv_results['test_accuracy'].mean()],
            'Recall': [cv_results['test_recall'].mean()],
            'F1 Score': [cv_results['test_f1'].mean()],
            'AUC': [cv_results['test_roc_auc'].mean()]
        })
        st.dataframe(score_df.T.rename(columns={0: "Score moyen"}).style.format("{:.4f}"))

# 6. Explication des Prédictions avec SHAP
elif page == "Explication des prédictions":
    st.title("🔍 Explication des Prédictions")

    # Vérification que X_scaled est défini correctement
    if 'X_scaled' not in locals():
        st.error("Les données n'ont pas été préalablement traitées et normalisées.")
    else:
        X_final_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Explainer SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_final_df)

        observation_idx = st.number_input(
            "Choisissez un index client à analyser", 
            min_value=0, 
            max_value=len(X_final_df) - 1, 
            step=1
        )

        prediction = model.predict([X_final.iloc[observation_idx]])

        prediction_label = "❌ Résilie" if prediction[0] == 1 else "✅ Ne résilie pas"
        st.markdown(f"### Pour l'Observation {observation_idx + 1}, le modèle a prédit que le client : **{prediction_label}**")

        # Affichage des valeurs SHAP
        st.markdown("#### Impact des variables :")
        for feature, shap_value in zip(X_final.columns, shap_values[1][observation_idx]):
            direction = "augmente" if shap_value > 0 else "diminue"
            st.markdown(
                f"- **{feature}** : La valeur SHAP est **{shap_value:+.4f}**, ce qui indique que la variable **{feature}** {'augmente' if shap_value > 0 else 'diminue'} la probabilité de résiliation."
            )

        # Graphique SHAP interactif
        shap.initjs()
        st.pydeck_chart(shap.force_plot(explainer.expected_value[1], shap_values[1][observation_idx], X_final.iloc[observation_idx]))

# 7. Personnalisation de l'interface
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f4f9;
        }
        .css-1v3fvcr {
            background-color: #3b8d99;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("**🔧 Paramètres du modèle :**")
# Exemples d'ajout de contrôle des paramètres du modèle
max_depth = st.sidebar.slider("Profondeur maximale de l'arbre", 1, 20, 3)
min_samples_leaf = st.sidebar.slider("Nombre minimum d'échantillons par feuille", 1, 20, 10)

