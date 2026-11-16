"""
Exporte le modèle champion dans le dossier model/
pour l'utiliser dans l'image Docker.
"""
import mlflow
import os
import shutil

# Configuration
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))

model_name = "churn-model"
model_uri = f"models:/{model_name}@champion"
export_path = "model"

try:
    # Suppression du dossier existant
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    
    # Téléchargement du modèle
    print(f"Téléchargement du champion depuis {model_uri}...")
    mlflow.artifacts.download_artifacts(model_uri, dst_path=export_path)
    
    print(f"✓ Champion exporté dans {export_path}/")
    print(f"  Prêt pour : docker build -t churn-api:1.0 .")
    
except Exception as e:
    print(f"✗ Erreur : {e}")
    print("  Vérifier que le modèle champion existe (register_model.py)")
