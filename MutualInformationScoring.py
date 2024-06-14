import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

# Load the preprocessed combined dataset
combined_df = pd.read_pickle('combined_df.pkl')

# Correct the column name to include the newline character
chosen_attributes = [
    'PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value',
    'Density\n(P/Km2)',
    'Agricultural Land( %)', 'Birth Rate', 'Co2-Emissions',
    'GDP', 'Infant mortality', 'Life expectancy', 'Out of pocket health expenditure',
    'Physicians per thousand', 'Population: Labor force participation (%)', 'Urban_population'
]

# Drop all rows with NaN values in the dataset
combined_df = combined_df.dropna(subset=chosen_attributes + ['Chronic Respiratory Diseases'])

# Splitting the dataset, using only the chosen attributes
X = combined_df[chosen_attributes]
y = combined_df['Chronic Respiratory Diseases']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use mutual_info_regression for continuous targets
selector = SelectKBest(score_func=mutual_info_regression, k='all').fit(X_train, Y_train)

# Getting the scores and sorting them to find the top features
scores = selector.scores_
indices = scores.argsort()[::-1]  # sort the scores in descending order

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), scores[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Feature Importance Using Mutual Information')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
