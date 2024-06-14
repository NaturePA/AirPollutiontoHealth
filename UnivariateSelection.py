import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
combined_df = pd.read_pickle('combined_df.pkl')

# Define your target variable and features list here
target_variable = 'Chronic Respiratory Diseases'  # Adjust as necessary
features = ['PM2.5 AQI Value', 'GDP', 'Urban_population', 'Co2-Emissions', 'Life expectancy']  # Example features

# Select relevant columns and drop rows with any missing values in them
df_filtered = combined_df[features + [target_variable]].dropna()

# Splitting the dataset into features (X) and target (y)
X = df_filtered[features]
y = df_filtered[target_variable]

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying Univariate Selection with SelectKBest using ANOVA F-value
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train, Y_train)

# Visualizing the feature scores
scores = selector.scores_
indices = np.argsort(scores)[::-1]  # Sorting the scores in descending order

# Plotting feature scores
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), scores[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title('Feature Importance Using Univariate Selection (ANOVA F-value)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()