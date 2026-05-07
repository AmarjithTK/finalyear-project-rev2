from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure upload and plots folders exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    from app.routes import app as routes_blueprint
    app.register_blueprint(routes_blueprint)
    
    return app