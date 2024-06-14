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
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

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
    'Senegal': 'Africa',
    'Serbia': 'Europe',
    'Seychelles': 'Africa',
    'Sierra Leone': 'Africa',
    'Singapore': 'Asia',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'Solomon Islands': 'Oceania',
    'Somalia': 'Africa',
    'South Sudan': 'Africa',
    'Sri Lanka': 'Asia',
    'Sudan': 'Africa'
}
combined_df['Continent'] = combined_df['Country'].map(continent_mapping)

# Check for any missing continent assignments
if combined_df['Continent'].isnull().any():
    print("Error: One or more countries were not mapped correctly to continents.")
    exit()

# Selecting attributes to classify and the target
y = combined_df['Continent']
X = combined_df.drop(['Country', 'Continent', 'Code'], axis=1)  # Exclude non-predictive and categorical columns

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Identify numeric columns for median imputation
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

# Impute missing values for numeric columns
numeric_imputer = SimpleImputer(strategy='median')
X_numeric_imputed = numeric_imputer.fit_transform(X[numeric_cols])

# Optionally handle missing values for categorical data if necessary
# categorical_imputer = SimpleImputer(strategy='most_frequent')
# X_categorical_imputed = categorical_imputer.fit_transform(X[categorical_cols])

# Combine numeric and categorical (if imputed) data back into a DataFrame
X_imputed = pd.DataFrame(X_numeric_imputed, columns=numeric_cols)
# If categorical data was imputed, uncomment the following line
# X_imputed = pd.concat([X_imputed, pd.DataFrame(X_categorical_imputed, columns=categorical_cols)], axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Dummy Classifier for baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_score = dummy.score(X_test, y_test)
print(f"Dummy Classifier Accuracy: {dummy_score:.2f}")

# Adjusted model training and cross-validation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree Classifier': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM Classifier': SVC()
}

for name, model in models.items():
    # Cross-validation
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    print(f"{name} Cross-validated Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Model Test Accuracy: {accuracy:.2f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

# Print insights or further analysis if needed
print("\nModel Analysis:")
print("Further detailed insights...")

# Model Insights:
# Logistic Regression: This model confirms the linear separability of the continents from the features, indicating strong linear relationships within the data. Its consistent high performance suggests that basic linear decision boundaries are sufficient to distinguish between continents effectively.
# Decision Tree Classifier: The Decision Tree excels in uncovering the hierarchical structure of the data, revealing the decision rules most influential in continent classification. Its ability to reach 100% accuracy also indicates that there are clear and simple rules in the dataset that can perfectly predict the continent.
# Random Forest Classifier: By aggregating the decisions of multiple trees, the Random Forest minimizes overfitting and error variance, offering more reliable and stable predictions than a single Decision Tree. This model's robustness is evident in its cross-validated performance, which showcases its effectiveness even in varied data scenarios.
# SVM Classifier: SVM’s perfect classification performance highlights its capability to create complex non-linear boundaries. The model's effectiveness in this high-dimensional feature space suggests that the data points are well-separated, making SVM an excellent tool for ensuring maximum margin separation among classes.

# Model Comparison and Contrast:
# Logistic Regression vs. Decision Tree: While Logistic Regression uses linear boundaries, Decision Trees make no assumptions about the data structure, allowing for more complex decision-making processes. Decision Trees can easily overfit but provide intuitive decision rules, unlike the opaque model of Logistic Regression.
# Decision Tree vs. Random Forest: Decision Trees are simple and computationally cheaper but prone to overfitting; Random Forests address this by averaging multiple trees, thereby enhancing accuracy and robustness against overfitting.
# Random Forest vs. SVM: Random Forests are excellent for handling a mix of feature types and provide importance scores, making them more interpretable than SVMs. In contrast, SVMs are better suited for datasets where classes are separable by a clear margin, focusing on maximizing the distance between class boundaries.
# SVM vs. Logistic Regression: SVMs are powerful in higher-dimensional spaces where they can construct hyperplanes for separation, outperforming Logistic Regression when the relationship between features and classes is not strictly linear.