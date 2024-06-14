import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# Load the preprocessed combined dataset again for ANOVA script
combined_df = pd.read_pickle('combined_df.pkl')

# Preparing the dataset: keep only numeric columns and drop rows with NaN values
numeric_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns
combined_df_numeric = combined_df[numeric_cols].dropna()

# Splitting the dataset into features and target
X = combined_df_numeric.drop('Chronic Respiratory Diseases', axis=1)
y = combined_df_numeric['Chronic Respiratory Diseases']

# Splitting into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANOVA F-value for feature selection
selector_anova = SelectKBest(score_func=f_regression, k='all').fit(X_train, Y_train)
scores_anova = selector_anova.scores_
indices_anova = scores_anova.argsort()[::-1]  # sort the scores in descending order

# Plotting the feature importances
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices_anova)), scores_anova[indices_anova])
plt.yticks(range(len(indices_anova)), [X.columns[i] for i in indices_anova])
plt.title('Feature Importance Using ANOVA F-value')
plt.xlabel('F-value Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
