'''
Name: Brian Arango
Date: 3/17/2023
Assignment: Module 9: Project - Part 1
Due Date: 3/17/2023
About this project: Join and analyze datasets to answer questions and perform data wrangling, scores and rankings, text attributes, and variance, covariance, and correlation analysis.
Assumptions: NA
All work below was performed by Brian Arango
Datasets are cause_of_deaths.csv, global_air_pollution_dataset.csv, and world-data-2023.csv
Dataset URLS:https://www.kaggle.com/datasets/iamsouravbanerjee/cause-of-deaths-around-the-world, https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset, https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023
Questions:
How do environmental quality and socioeconomic factors influence the prevalence of respiratory diseases across different countries?
What is the relationship between air pollution levels and the incidence of chronic diseases across various socioeconomic strata?
Answers are provided by correlations output of the code.
''' 
import pandas as pd
from scipy.stats import gmean
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load datasets
cause_of_deaths_df = pd.read_csv('cause_of_deaths.csv')
global_air_pollution_df = pd.read_csv('global_air_pollution_dataset.csv')
world_data_df = pd.read_csv('world-data-2023.csv')

def clean_data(df):
    for column in df.columns:
        if df[column].dtype == object:
            # Remove non-numeric characters except for the decimal point
            cleaned_series = df[column].str.replace('[^\d.]', '', regex=True)
            cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')
            
            # Check if the cleaned series doesn't result in all NaN values
            if not cleaned_series.isnull().all():
                df[column] = cleaned_series
            else:
                print(f"Column '{column}' could not be converted to numeric without losing all data.")
    return df

cause_of_deaths_df = clean_data(cause_of_deaths_df)
global_air_pollution_df = clean_data(global_air_pollution_df)
world_data_df = clean_data(world_data_df)

# Standardize country names for merging
cause_of_deaths_df.rename(columns={'Country/Territory': 'Country'}, inplace=True)
world_data_df.rename(columns={'Country': 'Country/Territory'}, inplace=True)
world_data_df.rename(columns={'Country/Territory': 'Country'}, inplace=True)

# Aggregate air pollution data by country
# Here, we average the AQI values for simplicity. Adjust the aggregation as needed for your analysis.
air_pollution_aggregated = global_air_pollution_df.groupby('Country').agg({
    'PM2.5 AQI Value': 'mean',
    'NO2 AQI Value': 'mean',
    'CO AQI Value': 'mean',
    'Ozone AQI Value': 'mean'
}).reset_index()

# Merge datasets on 'Country'
combined_df = pd.merge(cause_of_deaths_df, world_data_df, on='Country', how='inner')
combined_df = pd.merge(combined_df, air_pollution_aggregated, on='Country', how='inner')

# Select attributes relevant to Question 1 and 2
attributes_q1 = [
    'Country', 'Year', 'Chronic Respiratory Diseases',
    'PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value',
    'Density\n(P/Km2)', 'Agricultural Land( %)', 'Birth Rate', 'Co2-Emissions',
    'GDP', 'Infant mortality', 'Life expectancy', 'Out of pocket health expenditure',
    'Physicians per thousand', 'Population: Labor force participation (%)', 'Urban_population'
]

# Adjust the attributes for Question 2 as needed
attributes_q2 = attributes_q1

# Function to analyze numeric attributes
def analyze_numeric_attributes(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        print(f"\nAnalysis of {column}:")
        print("Max:", df[column].max())
        print("Min:", df[column].min())
        print("Mean:", df[column].mean())
        print("Median:", df[column].median())
        try:
            print("Geometric Mean:", gmean(df[column].dropna()))
        except ValueError:
            print("Geometric Mean: Not calculable due to negative values")
        print("Standard Deviation:", df[column].std())

# Function to analyze non-numeric attributes
def analyze_non_numeric_attributes(df):
    for column in df.select_dtypes(exclude=['float64', 'int64']).columns:
        print(f"\nAnalysis of {column}:")
        print("Possible values and counts:")
        print(df[column].value_counts())

# Aggregation with air pollution data
print("Analysis for Question 1:")
df_q1 = combined_df[attributes_q1]  
analyze_numeric_attributes(df_q1)
analyze_non_numeric_attributes(df_q1)

# Question 2: What is the relationship between air pollution levels and the incidence of chronic diseases across various socioeconomic strata?1

attributes_q2 = [
    'Country', 'Year', 'Cardiovascular Diseases', 'Diabetes Mellitus', 'Chronic Kidney Disease',
    'PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value',
    'Density\n(P/Km2)', 'Agricultural Land( %)', 'Birth Rate', 'Co2-Emissions',
    'GDP', 'Infant mortality', 'Life expectancy', 'Out of pocket health expenditure',
    'Physicians per thousand', 'Population: Labor force participation (%)', 'Urban_population'
]

# Similar to Question 1, we now extract a DataFrame for Question 2 analysis
df_q2 = combined_df[attributes_q2]  # Ensure these attributes are correctly named and present

print("\nAnalysis for Question 2:")
analyze_numeric_attributes(df_q2)
analyze_non_numeric_attributes(df_q2)

# Data Wrangling & Scores and Rankings (30 points)

# Assuming 'combined_df' is your combined dataset already loaded
# Filter for Canada
canada_df = combined_df[combined_df['Country'] == 'Canada'].copy()

# Handle NA values by replacing them with the mean for 'Chronic Respiratory Diseases'
mean_chronic_resp_diseases = canada_df['Chronic Respiratory Diseases'].mean()
canada_df['Chronic Respiratory Diseases'] = canada_df['Chronic Respiratory Diseases'].fillna(mean_chronic_resp_diseases)

# Plotting the total number of deaths due to Chronic Respiratory Diseases over the years as a scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(canada_df['Year'], canada_df['Chronic Respiratory Diseases'], alpha=0.6, label='Number of Deaths')

plt.title('Chronic Respiratory Diseases Deaths in Canada Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()

# Text Attributes (30 points)

# Calculate the average number of deaths due to Chronic Respiratory Diseases for each country
avg_deaths_by_country = combined_df.groupby('Country')['Chronic Respiratory Diseases'].mean()

# Define categories based on distribution
def categorize_incidence(deaths):
    if deaths < avg_deaths_by_country.quantile(0.33):
        return 'Low'
    elif deaths < avg_deaths_by_country.quantile(0.66):
        return 'Medium'
    else:
        return 'High'

# Apply categorization
incidence_categories = avg_deaths_by_country.apply(categorize_incidence)

# Count the number of countries in each category
category_counts = incidence_categories.value_counts()

# Reorder the categories
categories_ordered = ['Low', 'Medium', 'High']
category_counts = category_counts.reindex(categories_ordered)

# Plot with adjusted y-axis limit
plt.figure(figsize=(10, 6))
bars = category_counts.plot(kind='bar', color='skyblue', zorder=3)

# Zoom in around the y-axis at the 50 mark
plt.ylim(0, 55)

# Set grid behind the bars and other elements
plt.grid(axis='y', zorder=0)

# Annotate the number on top of each bar
for bar in bars.patches:
    plt.annotate(format(bar.get_height(), '.1f'),
                 (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha='center', va='center',
                 size=10, xytext=(0, 8),
                 textcoords='offset points')

plt.title('Distribution of Countries by Incidence Rate Category for Chronic Respiratory Diseases')
plt.xlabel('Incidence Rate Category')
plt.ylabel('Number of Countries')
plt.xticks(rotation=0)  # Keep the category names horizontal for readability
plt.tight_layout()
plt.show()

#Variance, Covariance, and Correlation (40 points)

# X attributes selected for demonstration (Replace with your chosen attributes)
x_attributes = [
    'PM2.5 AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Ozone AQI Value',
    'Density\n(P/Km2)', 'Agricultural Land( %)', 'Birth Rate', 'Co2-Emissions',
    'GDP', 'Infant mortality', 'Life expectancy', 'Out of pocket health expenditure',
    'Physicians per thousand', 'Population: Labor force participation (%)', 'Urban_population'
]
y_attribute = 'Chronic Respiratory Diseases'

# Compute and display variance for each X attribute
variance_data = combined_df[x_attributes].var()
print("Variance of each X attribute:")
print(variance_data)

# Compute and display covariance between each X attribute and the Y attribute
covariance_data = combined_df[x_attributes + [y_attribute]].cov()[y_attribute][:-1]
print("\nCovariance between each X attribute and Y attribute:")
print(covariance_data)

# Description of covariance
for x_attr in x_attributes:
    cov = covariance_data[x_attr]
    if cov > 0:
        print(f"The covariance between {x_attr} and {y_attribute} is positive, indicating that as {x_attr} increases, {y_attribute} tends to increase as well.")
    elif cov < 0:
        print(f"The covariance between {x_attr} and {y_attribute} is negative, indicating that as {x_attr} increases, {y_attribute} tends to decrease.")
    else:
        print(f"The covariance between {x_attr} and {y_attribute} is zero, indicating no linear relationship between the two.")

# Compute and display correlation between each X attribute and the Y attribute
correlation_data = combined_df[x_attributes + [y_attribute]].corr()[y_attribute][:-1]
print("\nCorrelation between each X attribute and Y attribute:")
print(correlation_data)

# Description of correlation
for x_attr in x_attributes:
    corr = correlation_data[x_attr]
    if corr > 0:
        degree = 'strong' if corr > 0.5 else 'moderate' if corr > 0.3 else 'weak'
        print(f"The correlation between {x_attr} and {y_attribute} is positive and {degree}. As {x_attr} increases, there is a {degree} tendency for {y_attribute} to increase.")
    elif corr < 0:
        degree = 'strong' if corr < -0.5 else 'moderate' if corr < -0.3 else 'weak'
        print(f"The correlation between {x_attr} and {y_attribute} is negative and {degree}. As {x_attr} increases, there is a {degree} tendency for {y_attribute} to decrease.")
    else:
        print(f"The correlation between {x_attr} and {y_attribute} is zero, indicating no linear relationship between the two.")

# Identify and Mitigate Outliers:  (40 points)

X_attribute1 = 'PM2.5 AQI Value'
X_attribute2 = 'Co2-Emissions'
Y_attribute = 'Chronic Respiratory Diseases'

# Create initial boxplots with Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].boxplot(combined_df[X_attribute1].dropna(), vert=True)
axs[0].set_title(f'Boxplot of {X_attribute1}')
axs[0].set_ylabel(X_attribute1)

axs[1].boxplot(combined_df[X_attribute2].dropna(), vert=True)
axs[1].set_title(f'Boxplot of {X_attribute2}')
axs[1].set_ylabel(X_attribute2)

plt.tight_layout()
plt.show()

# Create initial scatter plots with Matplotlib
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(combined_df[X_attribute1], combined_df[Y_attribute], alpha=0.5)
axs[0].set_title(f'{Y_attribute} vs. {X_attribute1}')
axs[0].set_xlabel(X_attribute1)
axs[0].set_ylabel(Y_attribute)

axs[1].scatter(combined_df[X_attribute2], combined_df[Y_attribute], alpha=0.5)
axs[1].set_title(f'{Y_attribute} vs. {X_attribute2}')
axs[1].set_xlabel(X_attribute2)
axs[1].set_ylabel(Y_attribute)

plt.tight_layout()
plt.show()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Apply the function to each X attribute
combined_df = remove_outliers(combined_df, X_attribute1)
combined_df = remove_outliers(combined_df, X_attribute2)

# Recreate boxplots after outlier removal
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].boxplot(combined_df[X_attribute1].dropna(), vert=True)
axs[0].set_title(f'Boxplot of {X_attribute1} (After Outlier Removal)')
axs[0].set_ylabel(X_attribute1)

axs[1].boxplot(combined_df[X_attribute2].dropna(), vert=True)
axs[1].set_title(f'Boxplot of {X_attribute2} (After Outlier Removal)')
axs[1].set_ylabel(X_attribute2)

plt.tight_layout()
plt.show()

# Recreate scatter plots after outlier removal
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(combined_df[X_attribute1], combined_df[Y_attribute], alpha=0.5)
axs[0].set_title(f'{Y_attribute} vs. {X_attribute1} (After Outlier Removal)')
axs[0].set_xlabel(X_attribute1)
axs[0].set_ylabel(Y_attribute)

axs[1].scatter(combined_df[X_attribute2], combined_df[Y_attribute], alpha=0.5)
axs[1].set_title(f'{Y_attribute} vs. {X_attribute2} (After Outlier Removal)')
axs[1].set_xlabel(X_attribute2)
axs[1].set_ylabel(Y_attribute)

plt.tight_layout()
plt.show()

combined_df.to_pickle('combined_df.pkl')  # Save as a pickle file
