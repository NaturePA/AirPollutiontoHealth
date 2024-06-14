import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load your DataFrame
combined_df = pd.read_pickle('combined_df.pkl')  # Ensure the DataFrame is loaded

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Global Health and Environmental Analysis"),
    
    html.Div([
        html.Label("Select Countries:"),
        dcc.Checklist(
            id='country-checklist',
            options=[{'label': country, 'value': country} for country in combined_df['Country'].unique()],
            value=[combined_df['Country'].unique()[0]],  # Default value
            style={'height': '100px', 'overflowY': 'scroll', 'width': '100%', 'border': '1px solid #ddd'}
        ),
    ]),
    
    html.Div([
        html.Label("Select Visualization:"),
        dcc.Dropdown(
            id='visualization-dropdown',
            options=[
                {'label': 'CO2 Emissions vs. Life Expectancy', 'value': 'CO2_vs_LifeExp'},
                {'label': 'CO2 Emissions vs. Population', 'value': 'CO2_vs_Population'},
            ],
            value='CO2_vs_LifeExp'  # Default value
        ),
    ]),
    
    html.Div([
        html.Label("CO2 Emissions Range (metric tons per capita):"),
        dcc.RangeSlider(
            id='co2-range-slider',
            min=combined_df['Co2-Emissions'].min(),
            max=combined_df['Co2-Emissions'].max(),
            step=0.1,
            value=[combined_df['Co2-Emissions'].min(), combined_df['Co2-Emissions'].max()],
            marks={str(int(x)): str(int(x)) for x in range(int(combined_df['Co2-Emissions'].min()), int(combined_df['Co2-Emissions'].max()), int((combined_df['Co2-Emissions'].max() - combined_df['Co2-Emissions'].min()) / 10))},
        ),
    ], style={'padding': '20px 0'}),
    
    html.Div([
        html.Label("Filter by Life Expectancy Range:"),
        html.Div([
            "Min:",
            dcc.Input(id='min-life-exp', type='number', value=combined_df['Life expectancy'].min(), style={'margin': '0 10px'}),
            "Max:",
            dcc.Input(id='max-life-exp', type='number', value=combined_df['Life expectancy'].max(), style={'margin': '0 10px'}),
        ]),
    ]),
    
    dcc.Graph(id='correlation-graph'),
])

@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('country-checklist', 'value'),
     Input('visualization-dropdown', 'value'),
     Input('co2-range-slider', 'value'),
     Input('min-life-exp', 'value'),
     Input('max-life-exp', 'value')]
)
def update_figure(selected_countries, selected_visualization, co2_range, min_life_exp, max_life_exp):
    filtered_df = combined_df[
        (combined_df['Country'].isin(selected_countries)) & 
        (combined_df['Co2-Emissions'] >= co2_range[0]) & 
        (combined_df['Co2-Emissions'] <= co2_range[1]) &
        (combined_df['Life expectancy'] >= min_life_exp) & 
        (combined_df['Life expectancy'] <= max_life_exp)
    ]

    if selected_visualization == 'CO2_vs_LifeExp':
        fig = px.scatter(filtered_df, x='Co2-Emissions', y='Life expectancy', color='Country', title='CO2 Emissions vs. Life Expectancy')
    else:
        fig = px.scatter(filtered_df, x='Co2-Emissions', y='Population', color='Country', title='CO2 Emissions vs. Population')

    fig.update_layout(transition_duration=500)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
