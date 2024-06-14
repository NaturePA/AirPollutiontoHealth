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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Load the DataFrame
combined_df = pd.read_pickle('combined_df.pkl')

continent_mapping = {
    'Afghanistan': 'Asia',
    'Albania': 'Europe',
    'Algeria': 'Africa',
    'Andorra': 'Europe',
    'Angola': 'Africa',
    'Armenia': 'Asia',
    'Austria': 'Europe',
    'Azerbaijan': 'Asia',
    'Bangladesh': 'Asia',
    'Barbados': 'North America',
    'Belarus': 'Europe',
    'Belgium': 'Europe',
    'Belize': 'North America',
    'Benin': 'Africa',
    'Bhutan': 'Asia',
    'Bosnia and Herzegovina': 'Europe',
    'Botswana': 'Africa',
    'Bulgaria': 'Europe',
    'Burkina Faso': 'Africa',
    'Burundi': 'Africa',
    'Cambodia': 'Asia',
    'Cameroon': 'Africa',
    'Central African Republic': 'Africa',
    'Chad': 'Africa',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Comoros': 'Africa',
    'Costa Rica': 'North America',
    'Croatia': 'Europe',
    'Cuba': 'North America',
    'Cyprus': 'Europe',
    'Denmark': 'Europe',
    'Dominican Republic': 'North America',
    'Ecuador': 'South America',
    'El Salvador': 'North America',
    'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa',
    'Estonia': 'Europe',
    'Ethiopia': 'Africa',
    'Finland': 'Europe',
    'Gabon': 'Africa',
    'Georgia': 'Asia',
    'Ghana': 'Africa',
    'Greece': 'Europe',
    'Guatemala': 'North America',
    'Guinea': 'Africa',
    'Guinea-Bissau': 'Africa',
    'Guyana': 'South America',
    'Haiti': 'North America',
    'Honduras': 'North America',
    'Hungary': 'Europe',
    'Iceland': 'Europe',
    'Israel': 'Asia',
    'Jamaica': 'North America',
    'Jordan': 'Asia',
    'Kenya': 'Africa',
    'Kyrgyzstan': 'Asia',
    'Latvia': 'Europe',
    'Lebanon': 'Asia',
    'Lesotho': 'Africa',
    'Liberia': 'Africa',
    'Libya': 'Africa',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Madagascar': 'Africa',
    'Malawi': 'Africa',
    'Maldives': 'Asia',
    'Mali': 'Africa',
    'Malta': 'Europe',
    'Mauritius': 'Africa',
    'Mongolia': 'Asia',
    'Montenegro': 'Europe',
    'Morocco': 'Africa',
    'Mozambique': 'Africa',
    'Myanmar': 'Asia',
    'Namibia': 'Africa',
    'Nepal': 'Asia',
    'New Zealand': 'Oceania',
    'Nicaragua': 'North America',
    'Niger': 'Africa',
    'Nigeria': 'Africa',
    'Norway': 'Europe',
    'Oman': 'Asia',
    'Palau': 'Oceania',
    'Panama': 'North America',
    'Papua New Guinea': 'Oceania',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Philippines': 'Asia',
    'Portugal': 'Europe',
    'Qatar': 'Asia',
    'Romania': 'Europe',
    'Rwanda': 'Africa',
    'Saint Kitts and Nevis': 'North America',
    'Saint Lucia': 'North America',
    'Suriname': 'South America',
    'Sweden': 'Europe',
    'Switzerland': 'Europe',
    'Tajikistan': 'Asia',
    'Togo': 'Africa',
    'Trinidad and Tobago': 'North America',
    'Tunisia': 'Africa',
    'Turkmenistan': 'Asia',
    'Uganda': 'Africa',
    'Uruguay': 'South America',
    'Uzbekistan': 'Asia',
    'Vanuatu': 'Oceania',
    'Yemen': 'Asia',
    'Zambia': 'Africa',
    'Zimbabwe': 'Africa',
}

combined_df['Continent'] = combined_df['Country'].map(continent_mapping)

# Feature Engineering
combined_df['Country_NameLength'] = combined_df['Country'].apply(len)

# One Hot Encoding for continents
combined_df = pd.concat([combined_df, pd.get_dummies(combined_df['Continent'], prefix='Continent')], axis=1)

# Count Encoding for continents
continent_counts = combined_df['Continent'].value_counts()
combined_df['Continent_Count'] = combined_df['Continent'].map(continent_counts)

# Standard Scaling for GDP
scaler = StandardScaler()
combined_df['GDP_Scaled'] = scaler.fit_transform(combined_df[['GDP']])

# Impute missing values
imputer = SimpleImputer(strategy='median')
features = ['GDP_Scaled', 'Population', 'Continent_Count', 'Country_NameLength']
combined_df[features + ['Co2-Emissions']] = imputer.fit_transform(combined_df[features + ['Co2-Emissions']])

# Prepare the data for modeling
X = combined_df[features]  # Predictor variables
y = combined_df['Co2-Emissions']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models with parameters adjusted to control for overfitting
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Dictionary to store test RMSEs
results = {}

# Train models and evaluate with cross-validation
for name, model in models.items():
    # Fit model to the training data
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse  # Store RMSE in the dictionary

# Analysis of the results
for model_name, rmse in results.items():
    if model_name == 'Linear Regression':
        print(f"{model_name} shows a simple linear relationship between predictors and CO2 emissions. The RMSE of {rmse} indicates the model's accuracy.")
    elif model_name == 'Decision Tree':
        print(f"{model_name} explores non-linear relationships and interactions between features. It offers a more nuanced understanding with an RMSE of {rmse}.")
    elif model_name == 'Random Forest':
        print(f"{model_name} builds on decision trees to reduce variance, offering more reliable predictions. Its RMSE of {rmse} reflects this improvement.")
    elif model_name == 'Gradient Boosting':
        print(f"{model_name} incrementally improves weak predictions, focusing on reducing bias and variance, as evidenced by its RMSE of {rmse}.")


# Model Comparison and Contrast
# Linear Regression vs Decision Tree: Linear Regression assumes a linear relationship between predictors and the target, making it less flexible for complex patterns compared to the Decision Tree, which can model non-linear relationships but may overfit if not properly tuned.
# Decision Tree vs Random Forest: While a single Decision Tree can quickly overfit to the training data, Random Forest mitigates this by averaging multiple decision trees, which generally improves the predictive performance and robustness.
# Random Forest vs Gradient Boosting: Both are ensemble techniques, but Random Forest builds each tree independently while Gradient Boosting builds one tree at a time in a sequential manner, focusing on correcting the previous trees' errors, often leading to better performance at the cost of increased computational complexity.
# Gradient Boosting vs Linear Regression: Gradient Boosting can model complex datasets by optimizing on loss functions and handles a variety of data types well, unlike Linear Regression which can only model linear relationships and may underperform on datasets with non-linear patterns.
