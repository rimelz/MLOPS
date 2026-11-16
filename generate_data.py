"""
Génère un jeu de données fictif pour la prédiction du churn client.
Crée 500 clients avec des caractéristiques aléatoires.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

n_clients = 500

# Génération des données
data = {
    'age': np.random.randint(18, 70, n_clients),
    'salaire': np.random.randint(20000, 100000, n_clients),
    'abonnement': np.random.choice(['Basic', 'Premium', 'Gold'], n_clients),
    'anciennete': np.random.randint(0, 10, n_clients),
    'consommation': np.random.randint(0, 100, n_clients)
}

df = pd.DataFrame(data)

# Règle de churn : churn = 1 si consommation < 30 et anciennete < 2
df['churn'] = ((df['consommation'] < 30) & (df['anciennete'] < 2)).astype(int)

# Sauvegarde
df.to_csv('churn_data.csv', index=False)
print(f"✓ Données générées : {len(df)} clients dans churn_data.csv")
print(f"  - Churn : {df['churn'].sum()} clients ({df['churn'].mean()*100:.1f}%)")
