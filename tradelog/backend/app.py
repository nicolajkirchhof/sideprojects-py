from flask import Flask, jsonify
from flask_migrate import Migrate
from flask_cors import CORS
from .config import Config
from .extensions import db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    Migrate(app, db)
    CORS(app)

    with app.app_context():
        from .models import Trade, TradeLog, TradeIdea, Position

    @app.route('/')
    def index():
        return jsonify({'message': 'Hello, World!'})

    @app.route('/trade-ideas')
    def trade_ideas():
        ideas = models.TradeIdea.query.all()
        return jsonify([idea.to_dict() for idea in ideas])

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
