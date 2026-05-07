from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import os
from app.utils.plotter import plot_data

app = Blueprint('app', __name__)

UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'plots'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return redirect(url_for('app.visualize', filename=file.filename))
    return render_template('index.html')

@app.route('/visualize/<filename>', methods=['GET', 'POST'])
def visualize(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)
    columns = df.columns.tolist()
    
    if request.method == 'POST':
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column')
        plot_file = plot_data(filepath, x_column, y_column)
        return redirect(url_for('app.show_plot', filename=os.path.basename(plot_file)))
    
    return render_template('visualize.html', columns=columns, filename=filename)

@app.route('/plot/<filename>')
def show_plot(filename):
    plots_dir = os.path.abspath(PLOTS_FOLDER)
    return send_from_directory(plots_dir, filename)