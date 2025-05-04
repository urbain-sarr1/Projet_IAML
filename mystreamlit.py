import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Charger vos données ici
df = pd.read_csv("churn_clients.csv")

X = df.drop(columns=['Resilie'])
y = df['Resilie']

# Diviser les données en ensemble d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialiser l'arbre de décision
model = DecisionTreeClassifier(random_state=42)

# Fonction pour calculer les scores de validation croisée
def get_cross_val_scores():
    scores_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    scores_recall = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
    scores_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    scores_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    st.write(f"**Moyenne de l'Accuracy (Validation Croisée)** : {scores_accuracy.mean():.4f}")
    st.write(f"**Moyenne du Recall (Validation Croisée)** : {scores_recall.mean():.4f}")
    st.write(f"**Moyenne du F1 Score (Validation Croisée)** : {scores_f1.mean():.4f}")
    st.write(f"**Moyenne du ROC AUC (Validation Croisée)** : {scores_auc.mean():.4f}")

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Afficher les indicateurs de performance sur l'ensemble de test
st.write(f"**Accuracy sur Test Set** : {accuracy_score(y_test, y_pred):.4f}")
st.write(f"**Recall sur Test Set** : {recall_score(y_test, y_pred):.4f}")
st.write(f"**F1 Score sur Test Set** : {f1_score(y_test, y_pred):.4f}")
st.write(f"**ROC AUC sur Test Set** : {roc_auc_score(y_test, y_pred):.4f}")

# Afficher l'importance des variables
st.subheader("Importance des Variables")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Créer un DataFrame pour l'affichage
feature_importance_df = pd.DataFrame({
    'Feature': [f"Feature {i+1}" for i in range(X.shape[1])],
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.write(feature_importance_df)

# Visualiser les importances des caractéristiques
fig, ax = plt.subplots()
sns.barplot(x=importances[indices], y=[f"Feature {i+1}" for i in indices], ax=ax)
ax.set_title('Importance des Variables')
st.pyplot(fig)

# Fonction pour afficher la prédiction et son explication
def explain_prediction(user_id):
    user_data = X_test.iloc[user_id].values.reshape(1, -1)
    user_pred = model.predict(user_data)[0]

    st.write(f"**Prédiction pour l'utilisateur {user_id}:** {user_pred}")
    
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_data)
    
    # Visualisation SHAP
    st.subheader(f"Explication de la Prédiction pour l'Utilisateur {user_id}")
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], user_data[0], matplotlib=True)
    st.pyplot()

# Interface pour choisir un utilisateur
st.sidebar.subheader("Choisir un utilisateur")
user_id = st.sidebar.slider("Sélectionner l'ID de l'utilisateur", 0, len(X_test) - 1, 0)
if st.sidebar.button("Afficher la Prédiction et son Explication"):
    explain_prediction(user_id)

# Afficher les scores de validation croisée
if st.button("Afficher les Scores de Validation Croisée"):
    get_cross_val_scores()

