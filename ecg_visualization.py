# Import necessary libraries
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Load the dataset
file_path = 'dataset/JS00001_filtered.csv'  # Update this path as necessary
ecg_data = pd.read_csv(file_path)

# Define leads
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Initialize the Dash app
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='ECG Data Dashboard', style={'textAlign': 'center', 'color': '#007BFF'}),
    html.Div([
        html.Label('Select ECG Lead for Analysis:'),
        dcc.Dropdown(
            id='lead-dropdown',
            options=[{'label': 'All Leads', 'value': 'ALL'}] + [{'label': lead, 'value': lead} for lead in leads],
            value='ALL'  # Default value to show all leads
        )
    ], style={'width': '30%', 'margin': '0 auto', 'padding': '20px'}),
    dcc.Graph(id='ecg-data-visualization'),
], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})

# Define callback to update the figure based on the selected lead
@app.callback(
    Output('ecg-data-visualization', 'figure'),
    [Input('lead-dropdown', 'value')]
)
def update_figure(selected_lead):
    # Initialize the figure with subplots
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=("ECG Signal Over Time", "Histogram of Signal Amplitudes",
                                        "Scatter Plot: Lead I vs Lead II", "Rolling Average"),
                        vertical_spacing=0.1,
                        specs=[[{"type": "scatter"}], [{"type": "histogram"}], [{"type": "scatter"}], [{"type": "scatter"}]])

    # Conditionally display either all leads or just the selected lead
    if selected_lead == 'ALL':
        # Show all leads
        for lead in leads:
            fig.add_trace(go.Scatter(x=ecg_data['time'], y=ecg_data[lead], mode='lines', name=f'Lead {lead}'), row=1, col=1)
            fig.add_trace(go.Histogram(x=ecg_data[lead], name=f'Lead {lead}', opacity=0.75), row=2, col=1)
    else:
        # Show only the selected lead
        ecg_data['RollingAvg'] = ecg_data[selected_lead].rolling(window=100).mean()
        fig.add_trace(go.Scatter(x=ecg_data['time'], y=ecg_data[selected_lead], mode='lines', name=f'Lead {selected_lead}'), row=1, col=1)
        fig.add_trace(go.Histogram(x=ecg_data[selected_lead], name=f'Lead {selected_lead}', opacity=0.75), row=2, col=1)
        fig.add_trace(go.Scatter(x=ecg_data['time'], y=ecg_data['RollingAvg'], mode='lines', name=f'Rolling Average: Lead {selected_lead}'), row=4, col=1)

    # Common settings for all cases
    fig.update_traces(opacity=0.75, bingroup=1, row=2, col=1)
    fig.update_layout(barmode='overlay')
    fig.add_trace(go.Scatter(x=ecg_data['I'], y=ecg_data['II'], mode='markers', name='Lead I vs Lead II'), row=3, col=1)

    # Update the figure layout
    fig.update_layout(height=1600, title_text="Comprehensive ECG Data Analysis", showlegend=True)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)