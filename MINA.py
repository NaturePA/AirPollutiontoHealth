import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import matplotlib.pyplot as plt

# Load the preprocessed combined dataset
combined_df = pd.read_pickle('combined_df.pkl')

# Exclude non-numeric columns and drop rows with NaN values
numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns
combined_df_numeric = combined_df[numeric_cols].dropna()

# Splitting the dataset
X = combined_df_numeric.drop('Chronic Respiratory Diseases', axis=1)
y = combined_df_numeric['Chronic Respiratory Diseases']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mutual Information Regression
selector = SelectKBest(score_func=mutual_info_regression, k='all').fit(X_train, Y_train)
scores = selector.scores_
indices = scores.argsort()[::-1]

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), scores[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Feature Importance Using Mutual Information')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
