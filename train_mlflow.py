"""
Entraîne 3 modèles RandomForest avec différentes configurations
et journalise les expériences dans MLflow.
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

# Configuration MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
mlflow.set_experiment("churn-prediction")

# Chargement des données
df = pd.read_csv('churn_data.csv')

# Encodage de la colonne abonnement
le = LabelEncoder()
df['abonnement_encoded'] = le.fit_transform(df['abonnement'])

# Configuration des 3 essais
experiments = [
    {
        'name': 'Essai 1',
        'n_estimators': 100,
        'max_depth': 5,
        'features': ['age', 'salaire', 'abonnement_encoded', 'anciennete', 'consommation']
    },
    {
        'name': 'Essai 2',
        'n_estimators': 50,
        'max_depth': 3,
        'features': ['age', 'salaire', 'abonnement_encoded', 'anciennete', 'consommation']
    },
    {
        'name': 'Essai 3',
        'n_estimators': 100,
        'max_depth': 5,
        'features': ['age', 'abonnement_encoded', 'anciennete', 'consommation']  # sans salaire
    }
]

# Boucle d'entraînement
for exp in experiments:
    with mlflow.start_run(run_name=exp['name']):
        # Préparation des données
        X = df[exp['features']]
        y = df['churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraînement
        model = RandomForestClassifier(
            n_estimators=exp['n_estimators'],
            max_depth=exp['max_depth'],
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Journalisation
        mlflow.log_param("n_estimators", exp['n_estimators'])
        mlflow.log_param("max_depth", exp['max_depth'])
        mlflow.log_param("features", ','.join(exp['features']))
        mlflow.log_param("n_features", len(exp['features']))
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Enregistrer le modèle avec les types de confiance
        mlflow.sklearn.log_model(
            model, 
            "model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"✓ {exp['name']} : accuracy={accuracy:.2f}, f1={f1:.2f}")

print("\n✓ 3 runs créés dans l'expérience 'churn-prediction'")
print("  Lancer 'mlflow ui' pour voir les résultats")
