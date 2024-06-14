'''
Name: Brian Arango
Date: 4/21/2023
Assignment: Module 14: Project Part 3 
Due Date: 4/21/2023
About this project:  Build a prediction model to predict CO2 emissions based on various features. The dataset used for this project is the 'combined_df.pkl' file, which contains information about countries and their CO2 emissions. The project involves data preprocessing, feature engineering, model training, and evaluation. The final output will be an analysis of the results and a comparison of different models.
All work below was performed by Brian Arango
Dataset URLS:https://www.kaggle.com/datasets/iamsouravbanerjee/cause-of-deaths-around-the-world, https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset, https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023
'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the DataFrame
df = pd.read_pickle('combined_df.pkl')

# Feature Selection: Assume 'GDP' and 'Population' are the chosen features
features = df[['GDP', 'Population']].dropna()

# Data scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the best number of clusters using the Elbow Method and Silhouette Score
sse = {}
silhouette_scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    sse[k] = kmeans.inertia_  # Sum of squared distances to closest cluster center
    silhouette_scores[k] = silhouette_score(scaled_features, kmeans.labels_)

# Plot SSE (Elbow Method)
plt.figure(figsize=(10, 6))
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.title('Elbow Method for Determining Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
plt.title('Silhouette Scores for Determining Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
