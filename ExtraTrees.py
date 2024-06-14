import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Load the dataset
combined_df = pd.read_pickle('combined_df.pkl')

# Selecting features and target, assuming 'Chronic Respiratory Diseases' is the target
features = ['PM2.5 AQI Value', 'GDP', 'Urban_population', 'Co2-Emissions', 'Life expectancy']
X = combined_df[features]
y = combined_df['Chronic Respiratory Diseases']

# Drop rows with NaN values
X_clean = X.dropna()
y_clean = y[X.index.isin(X_clean.index)]

# Fitting ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X_clean, y_clean)

# Visualizing feature importances
feat_importances = pd.Series(model.feature_importances_, index=X_clean.columns)
feat_importances.nlargest(len(features)).plot(kind='barh')
plt.title('Feature Importance with ExtraTreesClassifier')
plt.show()
