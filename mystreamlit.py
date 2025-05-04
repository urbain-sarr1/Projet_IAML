import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Charger les données
@st.cache_data
def load_and_preprocess_data():
    # Charger le fichier CSV ou Parquet
    df = pd.read_csv('churn_clients.csv')
    
    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    df['Sexe'] = label_encoder.fit_transform(df['Sexe'])
    df['Support_contacte'] = label_encoder.fit_transform(df['Support_contacte'])

    # Normalisation des colonnes numériques
    scaler = StandardScaler()
    df[['Age', 'Revenu', 'Anciennete', 'Frequence_utilisation', 'Score_satisfaction']] = scaler.fit_transform(
        df[['Age', 'Revenu', 'Anciennete', 'Frequence_utilisation', 'Score_satisfaction']])

    return df

df = load_and_preprocess_data()

# Séparer les données
X = df.drop('Resilie', axis=1)
y = df['Resilie']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Sélectionner les modèles
models = {
    'Régression Logistique': LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=42),
    'Arbre de Décision': DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=10, weights='distance')
}

# Afficher les graphiques de distribution
st.title("Analyse des clients et prédiction de la résiliation")
st.subheader("Distribution de l'âge")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("Distribution du revenu")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.histplot(df['Revenu'], kde=True, ax=ax2)
st.pyplot(fig2)

st.subheader("Matrice de corrélation")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# Sélection du modèle
model_name = st.selectbox("Sélectionner le modèle", options=list(models.keys()))
model = models[model_name]

# Pipeline avec SMOTE
pipeline = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)), ('model', model)])

# Validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Entraînement et évaluation sur les différentes métriques
st.subheader(f"Évaluation du modèle {model_name}")
metrics = ['Accuracy', 'Recall', 'F1 Score', 'AUC']

for metric in metrics:
    st.write(f"{metric} (cross-validation):")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric.lower())
    st.write(f"  - Moyenne : {scores.mean():.4f}")

# Entraînement final
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# Entraîner le modèle
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test_scaled)

# Évaluation finale sur le test set
st.subheader("Évaluation sur le test set")
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Précision : {accuracy:.4f}")
st.write(f"Rappel : {recall:.4f}")
st.write(f"F1 Score : {f1:.4f}")
st.write(f"AUC : {auc:.4f}")
st.write("Matrice de confusion :")
st.write(conf_matrix)

# Affichage de la courbe ROC
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
fig4, ax4 = plt.subplots(figsize=(6, 5))
ax4.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
ax4.plot([0, 1], [0, 1], 'k--')
ax4.set_xlabel('Taux de faux positifs')
ax4.set_ylabel('Taux de vrais positifs')
ax4.set_title('Courbe ROC')
ax4.legend(loc='best')
st.pyplot(fig4)

# Affichage de l'importance des variables
st.subheader(f"Importance des variables ({model_name})")

if model_name == 'Régression Logistique':
    importance = model.coef_[0]
elif model_name == 'Arbre de Décision':
    importance = model.feature_importances_
elif model_name == 'KNN':
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42, scoring='accuracy')
    importance = result.importances_mean

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

st.write(importance_df)

# Affichage SHAP pour l'Arbre de Décision
if model_name == 'Arbre de Décision':
    st.subheader("Explication SHAP pour l'Arbre de Décision")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    observation_idx = st.slider("Sélectionner un utilisateur pour l'explication SHAP", 0, X_test_scaled.shape[0] - 1)
    
    # Afficher la prédiction
    prediction = model.predict([X_test_scaled[observation_idx]])
    st.write(f"Prédiction pour l'utilisateur {observation_idx + 1}: {prediction[0]}")
    
    # Afficher les valeurs SHAP
    shap.initjs()
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][observation_idx], X_test.iloc[observation_idx]))

