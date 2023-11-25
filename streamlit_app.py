pip install dash
# Importing the necessary files
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
from dash import Dash, dcc, html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Data loading and preprocessing
stock_data = pd.read_csv("data.csv")
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'].str.replace(',', ''),
                                     errors='coerce')

# Convert 'Date' column to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

current_time_str = stock_data['Date'].max()
print(current_time_str)
current_time = current_time_str.to_pydatetime()
print(current_time)

# Create a dropdown filter for company selection
companies = stock_data['Name'].unique().tolist()

# Calculate VWAP for each company
stock_data['VWAP'] = (stock_data['Closing_Price'] * stock_data['Volume']
                      ).cumsum() / stock_data['Volume'].cumsum()

# Calculate Simple Moving Average (SMA) for Closing_Price (taking a window of 3 days as an example)
stock_data['SMA'] = stock_data['Closing_Price'].rolling(window=3).mean()

# Calculate Bollinger Bands
window = 20  # You can adjust the window size as needed
stock_data['Middle_Band'] = stock_data['Closing_Price'].rolling(
    window=window).mean()
stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()
stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()

# Calculate RSI for Closing_Price (taking a window of 14 days as an example)
delta = stock_data['Closing_Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Pivot the stock_dataset to have Date as the index and companies as columns
pivot_stock_data = stock_data.pivot(index='Date',
                                    columns='Name',
                                    values='Closing_Price')

# Calculate correlation matrix
correlation_matrix = pivot_stock_data.corr()

# Calculate quartiles for correlation coefficients
coefficients = correlation_matrix.values.flatten()
lower_quantile = np.percentile(coefficients, 25)
upper_quantile = np.percentile(coefficients, 75)

# Define contours for lower and upper quartiles
contour_lower = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=-1, end=lower_quantile,
                  size=0.05),  # Adjust size as needed
    colorscale='Blues',  # Choose a colorscale for lower quartile contours
    showscale=False,
    name=f'Lower Quartile Contours (-1 - {lower_quantile:.2f})')

contour_upper = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=upper_quantile, end=1,
                  size=0.05),  # Adjust size as needed
    colorscale='Reds',  # Choose a colorscale for upper quartile contours
    showscale=False,
    name=f'Upper Quartile Contours ({upper_quantile:.2f} - 1)')
# Create an empty figure
fig = go.Figure()
fig_rs = go.Figure()
fig_sm = go.Figure()
fig_cs = go.Figure()
fig_bc = go.Figure()
fig_ts = go.Figure()
fig2 = go.Figure()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define your layout for each visualization
# Visualization 1 layout
visualization_1_layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown_2',
        options=[{
            'label': company,
            'value': company
        } for company in stock_data['Name'].unique()],
        value=stock_data['Name'].iloc[0]  # Default value for dropdown
    ),
    dcc.Graph(figure=fig2, id='vwap-chart_1')
])

# Visualization 2 layout
ts_layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown_ts',
        options=[{
            'label': company,
            'value': company
        } for company in companies],
        value=companies[0]  # Set default value to the first company
    ),
    dcc.Graph(figure=fig_ts, id='line-chart')
])

# Visualization 3 layout
candlestick_layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown_cs',
        options=[{
            'label': company,
            'value': company
        } for company in companies],
        value=companies[0]  # Set default value to the first company
    ),
    dcc.Graph(figure=fig_cs, id='candlestick-chart')
])

# Visualization 4 layout
vwap_layout = html.Div([
    html.Label('Select Company:'),
    dcc.Dropdown(
        id='company-dropdown_v',
        options=[{
            'label': company,
            'value': company
        } for company in stock_data['Name'].unique()],
        value=stock_data['Name'].iloc[0]  # Default value for dropdown
    ),
    html.Br(),
    html.Label('Select Timeframe:'),
    dcc.RadioItems(id='timeframe-radio_v',
                   options=[{
                       'label': 'Last 1 Month',
                       'value': '1M'
                   }, {
                       'label': 'Last 5 Months',
                       'value': '5M'
                   }, {
                       'label': 'Last 1 Year',
                       'value': '1Y'
                   }, {
                       'label': 'Last 5 Years',
                       'value': '5Y'
                   }, {
                       'label': 'Entire Duration',
                       'value': 'All'
                   }],
                   value='All',
                   labelStyle={'display': 'block'}),
    dcc.Graph(id='vwap-chart')
])
# Visualization 5 layout
sma_layout = html.Div([
    html.Label('Select Company:'),
    dcc.Dropdown(
        id='company-dropdown_s',
        options=[{
            'label': company,
            'value': company
        } for company in stock_data['Name'].unique()],
        value=stock_data['Name'].iloc[0]  # Default value for dropdown
    ),
    html.Br(),
    html.Label('Select Timeframe:'),
    dcc.RadioItems(id='timeframe-radio_s',
                   options=[{
                       'label': 'Last 1 Month',
                       'value': '1M'
                   }, {
                       'label': 'Last 5 Months',
                       'value': '5M'
                   }, {
                       'label': 'Last 1 Year',
                       'value': '1Y'
                   }, {
                       'label': 'Last 5 Years',
                       'value': '5Y'
                   }, {
                       'label': 'Entire Duration',
                       'value': 'All'
                   }],
                   value='All',
                   labelStyle={'display': 'block'}),
    dcc.Graph(id='sma-chart')
])

# Visualization 6 layout
rsi_layout = html.Div([
    html.Label('Select Company:'),
    dcc.Dropdown(
        id='company-dropdown_r',
        options=[{
            'label': company,
            'value': company
        } for company in stock_data['Name'].unique()],
        value=stock_data['Name'].iloc[0]  # Default value for dropdown
    ),
    html.Br(),
    html.Label('Select Timeframe:'),
    dcc.RadioItems(id='timeframe-radio_r',
                   options=[{
                       'label': 'Last 1 Month',
                       'value': '1M'
                   }, {
                       'label': 'Last 5 Months',
                       'value': '5M'
                   }, {
                       'label': 'Last 1 Year',
                       'value': '1Y'
                   }, {
                       'label': 'Last 5 Years',
                       'value': '5Y'
                   }, {
                       'label': 'Entire Duration',
                       'value': 'All'
                   }],
                   value='All',
                   labelStyle={'display': 'block'}),
    dcc.Graph(figure=fig_rs, id='rsi-chart')
])

# Visualization 7 layout
correlation_heatmap_layout = html.Div([
    dcc.Graph(
        id='correlation-heatmap',
        figure={
            'data': [
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.index,
                    y=correlation_matrix.columns,
                    colorscale='Viridis',
                    colorbar=dict(title='Correlation'),
                ),
                contour_lower,  # Add lower quartile contours
                contour_upper  # Add upper quartile contours
            ],
            'layout': {
                'title': 'Correlation Heatmap with Quartile Contours',
                'xaxis': dict(title='Companies'),
                'yaxis': dict(title='Companies'),
                'paper_bgcolor': 'white',
                'font': dict(family='Arial', size=12, color='Black')
            }
        })
])

# Visualization 8 layout
bollinger_bands_layout = html.Div([
    html.Label('Select Company:'),
    dcc.Dropdown(
        id='company-dropdown_b',
        options=[{
            'label': company,
            'value': company
        } for company in stock_data['Name'].unique()],
        value=stock_data['Name'].iloc[0]  # Default value for dropdown
    ),
    html.Br(),
    html.Label('Select Timeframe:'),
    dcc.RadioItems(id='timeframe-radio_b',
                   options=[{
                       'label': 'Last 1 Month',
                       'value': '1M'
                   }, {
                       'label': 'Last 5 Months',
                       'value': '5M'
                   }, {
                       'label': 'Last 1 Year',
                       'value': '1Y'
                   }, {
                       'label': 'Last 5 Years',
                       'value': '5Y'
                   }, {
                       'label': 'Entire Duration',
                       'value': 'All'
                   }],
                   value='All',
                   labelStyle={'display': 'block'}),
    dcc.Graph(figure=fig_bc, id='bollinger-bands-chart')
])

# Combine all visualizations vertically
app.layout = html.Div([
    html.H1("Interactive charts"), visualization_1_layout, ts_layout,
    candlestick_layout, vwap_layout, sma_layout, rsi_layout,
    correlation_heatmap_layout, bollinger_bands_layout
])

# Define callback functions for interactivity
# Define callback to update the VWAP chart based on selected company
@app.callback(
    Output('vwap-chart_1', 'figure'),
    [Input('company-dropdown_2', 'value')]
)

def update_fig2_chart(selected_company):
    filtered_stock_data = stock_data[stock_data['Name'] == selected_company]
    fig2 = px.scatter(filtered_stock_data, x='Date', y='VWAP', title=f'Volume-Weighted Average Price (VWAP) for {selected_company}',
                     color='Closing_Price',  # Encode Volume information using color
                     size='Closing_Price',   # Encode Volume information using marker size
                     color_continuous_scale='Viridis',  # Choose a color scale
                     labels={'VWAP': 'VWAP', 'Closing_Price': 'Closing_Price'},  # Set labels for axes
                     hover_data={'Date': '|%B %d, %Y'})  # Format hover data for date
    return fig2

@app.callback(
  Output('line-chart', 'figure'),
  [Input('company-dropdown_ts', 'value')]
)
def update_ts_chart(selected_company):
  selected_company_data = stock_data[stock_data['Name'] == selected_company]

  fig_ts = go.Figure()

  for company in companies:
      if company == selected_company:
          fig_ts.add_trace(go.Scatter(x=selected_company_data['Date'],
                                   y=selected_company_data['Closing_Price'],
                                   mode='lines',
                                   name=company,
                                   visible=True))
      else:
          fig_ts.add_trace(go.Scatter(x=[],
                                   y=[],
                                   mode='lines',
                                   name=company,
                                   visible=False))

  fig_ts.update_layout(title='Interactive Time-Series Line Chart For Stock Closing Prices',
                    xaxis_title='Date',
                    yaxis_title='Closing Stock Price')

  return fig_ts

# CANDLE-STICK
# Iterate through each company to add candlestick traces to the figure
for company in companies:
  company_data = stock_data[stock_data['Name'] == company]
  fig_cs.add_trace(
      go.Candlestick(x=company_data['Date'],
                     open=company_data['Open'],
                     high=company_data['Daily_High'],
                     low=company_data['Daily_Low'],
                     close=company_data['Closing_Price'],
                     name=company,
                     visible=False))

# Set the visibility of the first company's trace to True initially
fig_cs.data[0].visible = True

# Create buttons for dropdown selection
buttons = []
for i, company in enumerate(companies):
  visibility = [i == j for j in range(len(companies))]
  button = dict(label=company,
                method="update",
                args=[{
                    "visible": visibility
                }, {
                    "title": f"Interactive Candlestick Chart - {company}"
                }])
  buttons.append(button)

# Update layout to include dropdown selection
fig_cs.update_layout(updatemenus=[{
    "buttons": buttons,
    "direction": "down",
    "showactive": True,
    "x": 0.5,
    "y": 1.5
}],
                  xaxis_title='Date',
                  yaxis_title='Price',
                  title='Interactive Candlestick Chart - Closing Prices')

# Define callback to update the candlestick chart based on dropdown selection
@app.callback(Output('candlestick-chart', 'figure'),
              [Input('company-dropdown_cs', 'value')])
def update_cs_chart(selected_company):
  # Create a copy of the figure to avoid modifying the original figure
  updated_fig_cs = fig_cs

  # Update visibility of the selected company's trace
  for trace in updated_fig_cs.data:
    trace.visible = (trace.name == selected_company)

  return updated_fig_cs

# Define callback to update the VWAP chart based on selected company and timeframe
@app.callback(Output('vwap-chart', 'figure'), [
    Input('company-dropdown_v', 'value'),
    Input('timeframe-radio_v', 'value')
])
def update_vwap_chart(selected_company, selected_timeframe):
  filtered_stock_data = stock_data[stock_data['Name'] == selected_company]

  if selected_timeframe == '1M':
    start_date = current_time - timedelta(days=30)
  elif selected_timeframe == '5M':
    start_date = current_time - timedelta(days=150)
  elif selected_timeframe == '1Y':
    start_date = current_time - timedelta(days=365)
  elif selected_timeframe == '5Y':
    start_date = current_time - timedelta(
        days=1825)  # Approximation of 5 years
  else:
    start_date = filtered_stock_data['Date'].min()  # Entire duration

  filtered_stock_data = filtered_stock_data[
      pd.to_datetime(filtered_stock_data['Date']) >= start_date]

  fig_vw = px.line(
      filtered_stock_data,
      x='Date',
      y='VWAP',
      title=f'Volume-Weighted Average Price (VWAP) for {selected_company}',
      line_shape='linear')  # Set line_shape to linear for continuous lines

  fig_vw.add_scatter(x=filtered_stock_data['Date'],
                     y=filtered_stock_data['Closing_Price'],
                     mode='markers',
                     marker=dict(color=filtered_stock_data['Closing_Price'],
                                 colorscale='Viridis',
                                 opacity=0.7,
                                 colorbar=dict(title='Closing Price'),
                                 line=dict(width=1, color='DarkSlateGrey')),
                     name='Price vs Volume')

  fig_vw.update_traces(
      line=dict(width=1),
      selector=dict(type='scatter'))  # Increase line width for main plot lines

  fig_vw.update_layout(
      xaxis=dict(title='Date',
                 tickformat='%Y-%m-%d',
                 showgrid=True,
                 linecolor='black',
                 linewidth=1),
      yaxis=dict(title='VWAP', showgrid=True, linecolor='black', linewidth=1),
      paper_bgcolor='white',  # Set paper (outside plot area) background color
      font=dict(family='Arial', size=10,
                color='Black')  # Set font style and size
  )
  return fig_vw


# Define callback to update the SMA chart based on selected company and timeframe
@app.callback(Output('sma-chart', 'figure'), [
    Input('company-dropdown_s', 'value'),
    Input('timeframe-radio_s', 'value')
])
def update_sma_chart(selected_company, selected_timeframe):
  filtered_stock_data = stock_data[stock_data['Name'] == selected_company]

  if selected_timeframe == '1M':
    start_date = current_time - timedelta(days=30)
  elif selected_timeframe == '5M':
    start_date = current_time - timedelta(days=150)
  elif selected_timeframe == '1Y':
    start_date = current_time - timedelta(days=365)
  elif selected_timeframe == '5Y':
    start_date = current_time - timedelta(
        days=1825)  # Approximation of 5 years
  else:
    start_date = filtered_stock_data['Date'].min()  # Entire duration

  filtered_stock_data = filtered_stock_data[
      pd.to_datetime(filtered_stock_data['Date']) >= start_date]

  fig_sm = px.line(filtered_stock_data,
                   x='Date',
                   y=['Closing_Price', 'SMA'],
                   title=f'Simple Moving Average (SMA) for {selected_company}')

  # Handle NaN values in 'SMA' column by replacing them with a default size
  sizes = filtered_stock_data['SMA'].fillna(
      10)  # Replace NaN values with default size 10

  # Add scatter plot for markers and channels
  fig_sm.add_scatter(x=filtered_stock_data['Date'],
                     y=filtered_stock_data['Closing_Price'],
                     mode='markers',
                     marker=dict(size=sizes,
                                 color=filtered_stock_data['SMA'],
                                 colorscale='Viridis',
                                 opacity=0.7,
                                 colorbar=dict(title='SMA'),
                                 line=dict(width=1, color='DarkSlateGrey')),
                     name='Price vs SMA')

  fig_sm.update_traces(
      mode='lines',
      selector=dict(type='scatter'))  # Set lines for main plot lines

  fig_sm.update_layout(
      xaxis=dict(title='Date',
                 tickformat='%Y-%m-%d',
                 showgrid=True,
                 linecolor='black',
                 linewidth=1),
      yaxis=dict(title='Price', showgrid=True, linecolor='black', linewidth=1),
      paper_bgcolor='white',  # Set paper (outside plot area) background color
      font=dict(family='Arial', size=12,
                color='Black')  # Set font style and size
  )
  return fig_sm

# Define callback to update the RSI chart based on selected company and timeframe
@app.callback(Output('rsi-chart', 'figure'), [
    Input('company-dropdown_r', 'value'),
    Input('timeframe-radio_r', 'value')
])
def update_rsi_chart(selected_company, selected_timeframe):
  filtered_stock_data = stock_data[stock_data['Name'] == selected_company]

  if selected_timeframe == '1M':
    start_date = current_time - timedelta(days=30)
  elif selected_timeframe == '5M':
    start_date = current_time - timedelta(days=150)
  elif selected_timeframe == '1Y':
    start_date = current_time - timedelta(days=365)
  elif selected_timeframe == '5Y':
    start_date = current_time - timedelta(
        days=1825)  # Approximation of 5 years
  else:
    start_date = filtered_stock_data['Date'].min()  # Entire duration

  filtered_stock_data = filtered_stock_data[
      pd.to_datetime(filtered_stock_data['Date']) >= start_date]

  # Calculate RSI for the filtered data
  delta = filtered_stock_data['Closing_Price'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  rsi = 100 - (100 / (1 + rs))

  # Create the figure for RSI chart using Plotly Express
  fig_rs = px.line(
      filtered_stock_data,
      x=filtered_stock_data['Date'],
      y=rsi,
      title=f'Relative Strength Index (RSI) for {selected_company}')

  # Add markers and channels for RSI visualization
  fig_rs.add_trace(
      go.Scatter(
          x=filtered_stock_data['Date'],
          y=rsi,
          mode='markers',  # Markers for RSI
          marker=dict(color=rsi,
                      colorscale='Viridis',
                      size=10,
                      colorbar=dict(title='RSI')),
          name='RSI Markers'))

  fig_rs.update_traces(line=dict(color='red',
                                 width=2))  # Update line color and width

  # Update layout
  fig_rs.update_layout(
      xaxis=dict(title='Date',
                 tickformat='%Y-%m-%d',
                 showgrid=True,
                 linecolor='black',
                 linewidth=1),
      yaxis=dict(title='RSI', showgrid=True, linecolor='black', linewidth=1),
      plot_bgcolor='white',  # Set plot background color
      paper_bgcolor='white',  # Set paper (outside plot area) background color
      font=dict(family='Arial', size=12,
                color='Black')  # Set font style and size
  )
  return fig_rs


# Define callback to update the Bollinger Bands chart based on selected company and timeframe
@app.callback(Output('bollinger-bands-chart', 'figure'), [
    Input('company-dropdown_b', 'value'),
    Input('timeframe-radio_b', 'value')
])
def update_bolinger_chart(selected_company, selected_timeframe):
  filtered_stock_data = stock_data[stock_data['Name'] == selected_company]

  if selected_timeframe == '1M':
    start_date = current_time - timedelta(days=30)
  elif selected_timeframe == '5M':
    start_date = current_time - timedelta(days=150)
  elif selected_timeframe == '1Y':
    start_date = current_time - timedelta(days=365)
  elif selected_timeframe == '5Y':
    start_date = current_time - timedelta(
        days=1825)  # Approximation of 5 years
  else:
    start_date = filtered_stock_data['Date'].min()  # Entire duration

  filtered_stock_data = filtered_stock_data[
      pd.to_datetime(filtered_stock_data['Date']) >= start_date]

  fig_bc = go.Figure()

  # Add Bollinger Bands lines
  fig_bc.add_trace(
      go.Scatter(x=filtered_stock_data['Date'],
                 y=filtered_stock_data['Middle_Band'],
                 mode='lines',
                 name='Middle_Band'))
  fig_bc.add_trace(
      go.Scatter(x=filtered_stock_data['Date'],
                 y=filtered_stock_data['Upper_Band'],
                 mode='lines',
                 name='Upper_Band'))
  fig_bc.add_trace(
      go.Scatter(x=filtered_stock_data['Date'],
                 y=filtered_stock_data['Lower_Band'],
                 mode='lines',
                 name='Lower_Band'))

  # Add marker for Closing_Price
  fig_bc.add_trace(
      go.Scatter(x=filtered_stock_data['Date'],
                 y=filtered_stock_data['Closing_Price'],
                 mode='markers',
                 name='Closing_Price',
                 marker=dict(color='red', size=8)))

  fig_bc.update_layout(
      title=f'Bollinger Bands for {selected_company}',
      xaxis=dict(title='Date',
                 tickformat='%Y-%m-%d',
                 showgrid=True,
                 linecolor='black',
                 linewidth=1),
      yaxis=dict(title='Price', showgrid=True, linecolor='black', linewidth=1),
      plot_bgcolor='white',  # Set plot background color
      paper_bgcolor='white',  # Set paper (outside plot area) background color
      font=dict(family='Arial', size=12,
                color='Black')  # Set font style and size
  )
  return fig_bc


# Run the app
if __name__ == '__main__':
  app.run_server(host='0.0.0.0', port=8080, debug=True)
