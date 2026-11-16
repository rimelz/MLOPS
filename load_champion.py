"""
Charge le modèle champion depuis le registre MLflow
et teste une prédiction.
"""
import mlflow
import pandas as pd
import os

# Configuration
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))

# Chargement du champion
model_name = "churn-model"
model_uri = f"models:/{model_name}@champion"

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✓ Champion chargé : {model_uri}")
    
    # Test de prédiction
    test_data = pd.DataFrame({
        'age': [25],
        'salaire': [35000],
        'abonnement_encoded': [0],
        'anciennete': [1],
        'consommation': [15]
    })
    
    prediction = model.predict(test_data)
    print(f"\nTest de prédiction :")
    print(f"  Client : age=25, salaire=35000, anciennete=1, consommation=15")
    print(f"  Prédiction : {'CHURN' if prediction[0] == 1 else 'NO CHURN'}")
    
except Exception as e:
    print(f"✗ Erreur : {e}")
    print("  Vérifier que le modèle est bien enregistré (register_model.py)")
