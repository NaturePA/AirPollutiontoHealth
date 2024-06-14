'''
Name: Brian Arango
Date: 3/24/2023
Assignment: Module 9: Project - Part 2
Due Date: 3/24/2023
About this project: Join and analyze datasets to answer questions and perform data wrangling, scores and rankings, text attributes, and variance, covariance, and correlation analysis.
Assumptions: NA
All work below was performed by Brian Arango
Datasets are cause_of_deaths.csv, global_air_pollution_dataset.csv, and world-data-2023.csv
Dataset URLS:https://www.kaggle.com/datasets/iamsouravbanerjee/cause-of-deaths-around-the-world, https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset, https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023
'''
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

combined_df = pd.read_pickle('combined_df.pkl')  # Load the pickle file

# Histogram: Population Density
histogram_fig = px.histogram(combined_df, x='Density\n(P/Km2)', title='Histogram of Population Density',
                             color_discrete_sequence=['#636EFA'])

# Boxplot: Life Expectancy
boxplot_fig = px.box(combined_df, y='Life expectancy', title='Boxplot of Life Expectancy',
                     color_discrete_sequence=['#EF553B'])

# Violin Plot: PM2.5 AQI Value
violin_fig = px.violin(combined_df, y='PM2.5 AQI Value', box=True, points='all', title='Violin Plot of PM2.5 AQI Value',
                       color_discrete_sequence=['#00CC96'])

# Scatter Plot: Chronic Respiratory Diseases vs. PM2.5 AQI Value
scatter_fig = px.scatter(combined_df, x='PM2.5 AQI Value', y='Chronic Respiratory Diseases', title='PM2.5 AQI vs. Chronic Respiratory Diseases',
                         color_continuous_scale=px.colors.sequential.Viridis)

co2_emissions_by_country = combined_df.groupby('Country')['Co2-Emissions'].mean().reset_index()

# Create a bar chart
bar_fig1 = px.bar(co2_emissions_by_country, x='Country', y='Co2-Emissions', title='CO2 Emissions by Country')

# Adjust layout for better readability
bar_fig1.update_layout(
    xaxis_title='Country',
    yaxis_title='CO2 Emissions',
    xaxis={'categoryorder':'total descending'},  # Optional: Sort countries by emissions
    xaxis_tickangle=-45  # Optional: Angle country names for better fit
)


# Bar Chart: Average Deaths by Country (Chronic Respiratory Diseases)
# Ensure you have a DataFrame named avg_deaths_by_country with 'Country' and 'Chronic Respiratory Diseases' columns for this chart
bar_fig = px.bar(combined_df.groupby('Country')['Chronic Respiratory Diseases'].mean().reset_index(), x='Country', y='Chronic Respiratory Diseases', title='Average Deaths by Country (Chronic Respiratory Diseases)',
                 color_continuous_scale=px.colors.sequential.Plasma)

# Initialize the Dash app
app = dash.Dash(__name__, assets_folder='assets')  # Ensure you have an 'assets' folder with your CSS

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1('Health and Environmental Analysis Dashboard', style={'textAlign': 'center'}),
    
    html.Div(children=[
        html.H2('Population Density Distribution'),
        dcc.Graph(figure=histogram_fig)
    ], style={'padding': 10}),
    
    html.Div(children=[
        html.H2('Life Expectancy Across Countries'),
        dcc.Graph(figure=boxplot_fig)
    ], style={'padding': 10}),
    
    html.Div(children=[
        html.H2('PM2.5 Air Quality Index Violin Plot'),
        dcc.Graph(figure=violin_fig)
    ], style={'padding': 10}),
    
    html.Div(children=[
        html.H2('Chronic Respiratory Diseases vs. PM2.5 AQI'),
        dcc.Graph(figure=scatter_fig)
    ], style={'padding': 10}),
    
    html.Div(children=[
        html.H2('CO2 Emissions by Country'),
        dcc.Graph(figure=bar_fig1)
    ], style={'padding': 10}),
    
    html.Div(children=[
        html.H2('Average Deaths by Country - Chronic Respiratory Diseases'),
        dcc.Graph(figure=bar_fig)
    ], style={'padding': 10})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
