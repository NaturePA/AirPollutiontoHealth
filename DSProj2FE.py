import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

# Load the DataFrame
combined_df = pd.read_pickle('combined_df.pkl')

unique_countries = combined_df['Country'].unique()
print(unique_countries)

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

# Map countries to continents
combined_df['Continent'] = combined_df['Country'].map(continent_mapping)

# Feature Engineering - Text Attributes
## Binary Feature
combined_df['Country_LongName'] = combined_df['Country'].apply(lambda x: 1 if len(x) > 5 else 0)

## Categorical Feature
# Already created 'Continent'

## Numeric Feature
combined_df['Country_NameLength'] = combined_df['Country'].apply(len)

# Encoding Categorical Text Attributes
## One Hot Encoding
combined_df = pd.concat([combined_df, pd.get_dummies(combined_df['Continent'], prefix='Continent')], axis=1)

## Count Encoding
continent_counts = combined_df['Continent'].value_counts()
combined_df['Continent_Count'] = combined_df['Continent'].map(continent_counts)

## Target Encoding
# Assuming 'Co2-Emissions' is a column for target encoding. This part requires careful handling to avoid data leakage.
# Normally, you would use a library like category_encoders or handle this with a more nuanced approach considering your modeling strategy.

# Feature Engineering - Numeric Attributes
## Binning
combined_df['Co2-Emissions_Binned'] = pd.cut(combined_df['Co2-Emissions'], bins=3, labels=['Low', 'Medium', 'High'])

## Standard Scaling
scaler = StandardScaler()
combined_df['GDP_Scaled'] = scaler.fit_transform(combined_df[['GDP']])

## Min-Max Scaling
min_max_scaler = MinMaxScaler()
combined_df['GDP_MinMax'] = min_max_scaler.fit_transform(combined_df[['GDP']])

## Quantile Transformation
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
combined_df['GDP_Quantile'] = quantile_transformer.fit_transform(combined_df[['GDP']])

# Plotly Charts
## Scatter Plot
scatter_fig = px.scatter(combined_df, x='GDP_Scaled', y='Co2-Emissions', color='Continent',
                         title='Scaled GDP vs CO2 Emissions by Continent')
scatter_fig.show()

## Bubble Plot
bubble_fig = px.scatter(combined_df, x='GDP_MinMax', y='Co2-Emissions', size='Population', color='Continent',
                        title='Min-Max Scaled GDP vs CO2 Emissions by Continent with Population Size')
bubble_fig.show()

## Heatmap to Show Correlation
corr = combined_df.select_dtypes(include=['float64', 'int']).corr()
heatmap_fig = go.Figure(data=go.Heatmap(
                   z=corr,
                   x=corr.columns,
                   y=corr.columns,
                   hoverongaps=False, colorscale='Viridis'))
heatmap_fig.update_layout(title='Correlation Heatmap')
heatmap_fig.show()