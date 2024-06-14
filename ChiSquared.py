import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

# Load the preprocessed combined dataset
combined_df = pd.read_pickle('combined_df.pkl')

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
combined_df['GDP_Cat'] = est.fit_transform(combined_df[['GDP']])
combined_df['Urban_population_Cat'] = est.fit_transform(combined_df[['Urban_population']])
combined_df['Chronic_Respiratory_Diseases_Cat'] = est.fit_transform(combined_df[['Chronic Respiratory Diseases']])

# Preparing the dataset for Chi-Squared feature selection
X_categorical = combined_df[['GDP_Cat', 'Urban_population_Cat']]
y_categorical = combined_df['Chronic_Respiratory_Diseases_Cat']

X_train_categorical, X_test_categorical, Y_train_categorical, Y_test_categorical = train_test_split(X_categorical, y_categorical, test_size=0.2, random_state=42)

# Chi-Squared feature selection
selector_categorical = SelectKBest(score_func=chi2, k='all').fit(X_train_categorical, Y_train_categorical)
scores_categorical = selector_categorical.scores_
indices_categorical = scores_categorical.argsort()[::-1]

# Plotting the Chi-Squared scores
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices_categorical)), scores_categorical[indices_categorical])
plt.yticks(range(len(indices_categorical)), [X_categorical.columns[i] for i in indices_categorical])
plt.title('Feature Importance Using Chi-Squared Test')
plt.xlabel('Chi-Squared Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
