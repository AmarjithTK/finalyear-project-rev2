import pandas as pd
import plotly.express as px
import os

def plot_data(file_path, x_column, y_column):
    df = pd.read_csv(file_path)
    
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("Selected columns are not in the DataFrame.")
    
    fig = px.line(df, x=x_column, y=y_column, title='Time Series Data')
    fig.update_xaxes(rangeslider_visible=True)
    
    plot_file_path = os.path.join('plots', 'microgrid_plot.html')
    fig.write_html(plot_file_path)
    
    return plot_file_path