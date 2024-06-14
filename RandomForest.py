import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the preprocessed combined dataset
combined_df = pd.read_pickle('combined_df.pkl')

# Select a subset of variables for the demonstration
features = ['PM2.5 AQI Value', 'GDP', 'Urban_population', 'Co2-Emissions', 'Life expectancy']
X = combined_df[features]
y = combined_df['Chronic Respiratory Diseases']  # Assuming this is correctly set for classification

# Drop rows with any missing values in X or y
complete_cases = pd.concat([X, y], axis=1).dropna()
X_clean = complete_cases[features]
y_clean = complete_cases['Chronic Respiratory Diseases']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# RandomForestClassifier to determine feature importance
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# Visualizing feature importances
feat_importances = pd.Series(model.feature_importances_, index=features)
feat_importances.nlargest(len(features)).plot(kind='barh')
plt.title('Feature Importance with RandomForestClassifier')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
