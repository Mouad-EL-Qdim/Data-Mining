import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données Zoo (assumant que vous avez les données chargées dans un DataFrame appelé 'zoo_data')
zoo_data = pd.read_csv('zoo/zoo.csv')

# Afficher les premières lignes du DataFrame
print(zoo_data.head())
zoo_data.head().to_excel('zoo//zoo_data_head.xlsx', index=False)

# Afficher les informations sur les colonnes et les types de données
print(zoo_data.info())

# Effectuer une analyse descriptive des données
print(zoo_data.describe())

describe_output = zoo_data.describe()
# Convertir la sortie en chaîne de caractères
describe_str = describe_output.to_string()
# Écrire la chaîne de caractères dans un fichier texte
with open('zoo/zoo_data_describe.txt', 'w') as file:
    file.write(describe_str)

# Visualiser la distribution des classes de l'attribut 'type'
sns.countplot(x='class_type', data=zoo_data)
plt.title('Distribution des classes')
plt.show()

# Visualiser la corrélation entre les attributs numériques
numeric_columns = zoo_data.select_dtypes(include='number')
correlation_matrix = numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()
