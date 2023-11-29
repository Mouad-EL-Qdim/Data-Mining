import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score, fowlkes_mallows_score

#    Charger la base de données Zoo sans l'attribut "type" :
# Charger les données Zoo
zoo_data = pd.read_csv("zoo/zoo.csv")

# Supprimer l'attribut "type"
zoo_data = zoo_data.drop("class_type", axis=1)

# Exclure la colonne 'animal_name'
zoo_data = zoo_data.drop("animal_name", axis=1)

#    Appliquer la Classification Ascendante Hiérarchique (CAH) :
# Calculer la matrice de liaison
Z = linkage(zoo_data, method="complete", metric='correlation')

# Afficher le dendrogramme
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=7)
plt.show()

#    Couper l'arbre obtenu pour obtenir 7 clusters :
# Couper l'arbre pour obtenir 7 clusters
clusters_cah = fcluster(Z, t=7, criterion="maxclust")
print(clusters_cah)
#    Appliquer le K-means clustering :
# Appliquer K-means avec k=7
kmeans = KMeans(n_clusters=7, random_state=6)
# mais par exemple si random_state=42 donc Accuracy (K-means) = 0.0
# donc il faut choisir une bonne valeur pour random_state
clusters_kmeans = kmeans.fit_predict(zoo_data) + 1
print(clusters_kmeans)
#    Comparer les clusters obtenus avec l'attribut "type" :
# Charger les labels "type"
labels = pd.read_csv("zoo/zoo.csv")["class_type"]

# Matrice de confusion pour CAH
cm_cah = confusion_matrix(labels, clusters_cah)

# Matrice de confusion pour K-means
cm_kmeans = confusion_matrix(labels, clusters_kmeans)

# Matrice de confusion pour CAH
# Convertir le tableau NumPy en DataFrame
cm_cah_df = pd.DataFrame(cm_cah)
print("cm_cah_df : \n", cm_cah_df)
# Sauvegarder le DataFrame dans un fichier Excel
cm_cah_df.to_excel('zoo/cm_cah.xlsx', index=False)

# Matrice de confusion pour K-means
# Convertir le tableau NumPy en DataFrame
cm_kmeans_df = pd.DataFrame(cm_kmeans)
print("\n cm_kmeans_df : \n", cm_kmeans_df)
# Sauvegarder le DataFrame dans un fichier Excel
cm_kmeans_df.to_excel('zoo/cm_kmeans.xlsx', index=False)

# Calcul de l'accuracy pour CAH
accuracy_cah = accuracy_score(labels, clusters_cah)

# Calcul de l'accuracy pour K-means
accuracy_kmeans = accuracy_score(labels, clusters_kmeans)

# Calcul de l'Adjusted Rand Index (ARI) pour CAH
ari_CAH = adjusted_rand_score(labels, clusters_cah)

# Calcul de l'Indice de Fowlkes-Mallows (FMI) pour CAH
fmi_CAH = fowlkes_mallows_score(labels, clusters_cah)

# Calcul de l'Adjusted Rand Index (ARI) pour K-means
ari_Km = adjusted_rand_score(labels, clusters_kmeans)

# Calcul de l'Indice de Fowlkes-Mallows (FMI) pour K-means
fmi_Km = fowlkes_mallows_score(labels, clusters_kmeans)

# Affichage des résultats
print("Accuracy (CAH):", accuracy_cah)
print("Accuracy (K-means):", accuracy_kmeans)
print("Adjusted Rand Index (ARI) [CAH]:", ari_CAH)
print("Fowlkes-Mallows Index (FMI) [CAH]:", fmi_CAH)
print("Adjusted Rand Index (ARI) [K-means]:", ari_Km)
print("Fowlkes-Mallows Index (FMI) [K-means]:", fmi_Km)
