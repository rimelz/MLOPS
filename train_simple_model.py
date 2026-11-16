"""
Script simple pour entraîner et sauvegarder model.pkl
Alternative au notebook pour générer rapidement le fichier .pkl
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Chargement des données
df = pd.read_csv('churn_data.csv')

# Encodage de l'abonnement
df['abonnement'] = df['abonnement'].map({
    'Basic': 0,
    'Standard': 1,
    'Premium': 2,
    'Gold': 1  # Au cas où
})

# Features et target - ESSAI 3 (sans salaire)
X = df[['age', 'abonnement', 'anciennete', 'consommation']]
y = df['churn']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entraînement - Essai 3 : 100 arbres, max_depth=5, sans salaire
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy Essai 3 (sans salaire) : {accuracy:.2f}")
print(f"Nombre de features : {model.n_features_in_}")

# Sauvegarde
joblib.dump(model, 'model.pkl')
print("✓ Modèle sauvegardé dans model.pkl")
