'''
Name: Brian Arango
Date: 4/21/2023
Assignment: Module 14: Project Part 3 
Due Date: 4/21/2023
About this project:  Build a clustering model to group countries based on GDP and Population. The dataset used for this project is the 'combined_df.pkl' file, which contains information about countries and their CO2 emissions. The project involves data preprocessing, feature selection, clustering with K-Means and Hierarchical Clustering, and analysis of the results. The final output will be a comparison and contrast of the clustering models.
All work below was performed by Brian Arango
Dataset URLS:https://www.kaggle.com/datasets/iamsouravbanerjee/cause-of-deaths-around-the-world, https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset, https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

# Load the DataFrame
df = pd.read_pickle('combined_df.pkl')

# Feature Selection
features = df[['GDP', 'Population']].dropna()

# Data scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Optimal number of clusters determined from previous script
optimal_k = 4  # Example based on previous results

# Clustering with K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Clustering with Agglomerative Hierarchical Clustering
agg_clust = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clust.fit_predict(scaled_features)

# Analysis of the results
print("K-Means Clustering Model:")
print(f"Clusters GDP and Population into {optimal_k} distinct groups, optimizing internal cohesion and separation among groups. This model efficiently partitions the dataset, which can help in identifying economic scales and population size groupings.")

print("Hierarchical Clustering Model:")
print(f"Groups data into a hierarchical structure of {optimal_k} clusters, potentially revealing multi-scale distribution patterns in GDP and Population data. This method is particularly useful for understanding nested dependencies and providing insights into hierarchical groupings within countries.")

# Comparison and Contrast of Clustering Models
# ------------------------------------------------------------------------------------------
# Both K-Means and Hierarchical Clustering are utilized to uncover distinct groups 
# within the dataset, based on GDP and Population, serving to structure complex datasets 
# effectively.
#
# K-Means Clustering is a centroid-based model that partitions data into a predetermined 
# number of clusters, making it efficient and scalable for large datasets. It requires 
# the number of clusters to be defined beforehand, which can be seen as a limitation if 
# the optimal cluster count is unknown.
#
# Hierarchical Clustering, in contrast, builds clusters by progressively merging or 
# splitting them based on distance, creating a dendrogram that illustrates data 
# hierarchies. This method does not necessitate a predefined cluster count, offering 
# advantages for exploratory analysis where data relationships are less defined, and 
# providing an intuitive view of cluster hierarchies.
# ------------------------------------------------------------------------------------------