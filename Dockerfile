# Image de base Python
FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Copie des dépendances
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du modèle exporté
COPY model/ ./model/

# Copie de l'application
COPY app.py .

# Variables d'environnement
ENV MODEL_URI=model
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
