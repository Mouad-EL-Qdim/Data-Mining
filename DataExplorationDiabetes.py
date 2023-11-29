import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données diabetes (assumant que vous avez les données chargées dans un DataFrame appelé 'diab_data')
diab_data = pd.read_csv('diabetes/diabetes.csv')

# Afficher les premières lignes du DataFrame
print(diab_data.head())
diab_data.head().to_excel('diabetes/diab_data_head.xlsx', index=False)

# Afficher les informations sur les colonnes et les types de données
print(diab_data.info())

# Effectuer une analyse descriptive des données
print(diab_data.describe())

# Effectuer une analyse descriptive des données
describe_output = diab_data.describe()
# Convertir la sortie en chaîne de caractères
describe_str = describe_output.to_string()
# Écrire la chaîne de caractères dans un fichier texte
with open('diabetes/diab_data_describe.txt', 'w') as file:
    file.write(describe_str)

# Ajouter la colonne 'type' basée sur la colonne 'Outcome'
diab_data['type'] = diab_data['Outcome'].apply(lambda x: 1 if x > 0 else 0)

# Convertir la colonne 'type' en variable catégorielle
diab_data['type'] = diab_data['type'].astype('category')

# Visualiser la distribution des classes de l'attribut 'type'
sns.countplot(x='type', data=diab_data)
plt.title('Distribution des classes')
plt.show()

# Visualiser la corrélation entre les attributs numériques
correlation_matrix = diab_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()
