from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree

# Charger les données Zoo
zoo_data = pd.read_csv("zoo/zoo.csv")

# Supprimer les lignes dupliquées
zoo_data = zoo_data.drop_duplicates()

# Convertir les fonctionnalités catégorielles en représentation numérique
zoo_data_encoded = pd.get_dummies(zoo_data)

# Diviser les données en attributs et classe
X = zoo_data_encoded.drop("class_type", axis=1)
y = zoo_data_encoded["class_type"]

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construire l'arbre de décision
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Prédire les classes pour l'ensemble de test avec l'arbre de décision
dt_y_pred = dt_clf.predict(X_test)

# Calculer la précision de l'arbre de décision
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print("Précision de l'arbre de décision :", dt_accuracy)

# Construire le classifieur KNN
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

# Prédire les classes pour l'ensemble de test avec KNN
knn_y_pred = knn_clf.predict(X_test)

# Calculer la précision de KNN
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("Précision de KNN :", knn_accuracy)

# Exemple de caractéristiques d'une tortue à prédire
tortoise_features = [0,0,1,0,0,0,0,0,1,1,0,0,4,1,0,1,3]

# Ajuster les caractéristiques de la tortue pour correspondre à la dimension attendue
if len(tortoise_features) < X.shape[1]:
    # Ajouter des colonnes remplies de zéros
    num_missing_features = X.shape[1] - len(tortoise_features)
    tortoise_features.extend([0] * num_missing_features)
elif len(tortoise_features) > X.shape[1]:
    # Supprimer les colonnes excédentaires
    tortoise_features = tortoise_features[:X.shape[1]]

# Prédire le type de la tortue
predicted_type = dt_clf.predict([tortoise_features])

# Afficher la prédiction
print("Le type de la tortue prédit est :", predicted_type)

# Générer l'arbre de décision
class_names = [str(c) for c in dt_clf.classes_]
feature_names = X_train.columns.astype(str)
tree.export_graphviz(dt_clf, out_file='zoo/treeDT.dot', feature_names=feature_names, class_names=class_names)
