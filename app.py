import os

from flask import Flask, render_template, request
from flask_cors import CORS

from routes.health import health_bp
from routes.query import query_bp
from routes.ready import ready_bp
from services.cache_service import is_cache_initialized, initialize_cache


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.before_request
    def check_cache():
        # Keep-alive / readiness endpoint should stay lightweight and never
        # trigger expensive initialization.
        if request.path == '/ready':
            return
        if not is_cache_initialized():
            initialize_cache()

    @app.route('/')
    def home():
        return render_template('index.html')

    app.register_blueprint(query_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(ready_bp)

    return app


app = create_app()


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n🌐 Starting server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)
