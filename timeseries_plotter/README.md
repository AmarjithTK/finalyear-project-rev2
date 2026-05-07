# Microgrid Visualizer

## Overview
Microgrid Visualizer is a web application that allows users to upload CSV files containing microgrid data, select from available tables, and visualize the data using interactive plots. The application is built with Flask and utilizes Plotly for data visualization.

## Features
- Upload CSV files containing microgrid data.
- Select specific tables from the uploaded data for visualization.
- Generate interactive line plots with a range slider for time series data.
- Save visualizations as HTML files.

## Project Structure
```
microgrid-visualizer
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── utils
│   │   └── plotter.py
│   ├── static
│   │   ├── css
│   │   │   └── styles.css
│   │   └── js
│   │       └── main.js
│   └── templates
│       ├── base.html
│       ├── index.html
│       └── visualize.html
├── uploads
│   └── .gitkeep
├── plots
│   └── .gitkeep
├── config.py
├── requirements.txt
├── run.py
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd microgrid-visualizer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application settings in `config.py` if necessary.

## Usage
1. Run the application:
   ```
   python run.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Use the upload form on the landing page to upload your CSV file.

4. Select the desired table from the uploaded data and submit the form to visualize the data.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.