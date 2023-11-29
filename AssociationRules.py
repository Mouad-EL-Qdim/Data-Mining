import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Charger la base de données Zoo
zoo_data = pd.read_csv('zoo/zoo.csv')

# Extraire les colonnes nécessaires pour l'analyse des règles d'association
association_data = zoo_data.drop('animal_name', axis=1)

# Supprimer les lignes dupliquées
association_data = association_data.drop_duplicates()

# Convertir les données en codage One-Hot
association_data = pd.get_dummies(association_data)
association_data = association_data.astype(bool)

# Extraire les itemsets fréquents avec un support minimum de 0.6
frequent_itemsets = apriori(association_data, min_support=0.6, use_colnames=True)
frequent_itemsets.to_excel('zoo/frequent_itemsets.xlsx', index=False)

# Générer les règles d'association avec une confiance minimale de 0.9
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)
filtered_rules = rules[rules['consequents'].astype(str).str.contains("class_type")]
filtered_rules.to_excel('zoo/SubRules.xlsx', index=False)

# Sélectionner les règles non redondantes
non_redundant_rules = filtered_rules[~filtered_rules.index.isin(filtered_rules['consequents'].apply(frozenset))]

# Visualiser les règles non redondantes
non_redundant_rules.to_excel('zoo/non_redundant_rules.xlsx', index=False)
