from flask import Flask
from flask_pymongo import PyMongo

mongo = PyMongo()

def create_app():
    app = Flask(__name__)
    app.config["MONGO_URI"] = "mongodb://localhost:27017/diagnosis_db"
    app.config["UPLOAD_FOLDER"] = "app/static/uploads"

    mongo.init_app(app)

    from .routes import routes
    app.register_blueprint(routes)

    return app
