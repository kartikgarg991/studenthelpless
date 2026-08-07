from flask import Blueprint, jsonify, make_response


ready_bp = Blueprint('ready', __name__)


@ready_bp.route('/ready', methods=['GET'])
def ready():
    resp = make_response(jsonify({'status': 'ok'}), 200)
    resp.headers['Cache-Control'] = 'no-store'
    return resp

