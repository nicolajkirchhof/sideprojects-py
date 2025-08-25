from flask import Blueprint, request, jsonify
from app.models import Position
from app import db

api_bp = Blueprint('api', __name__)

# [CREATE] - Add a new position
@api_bp.route('/positions', methods=['POST'])
def add_position():
  data = request.get_json()
  if not data or not all(k in data for k in ['ticker', 'strike', 'right', 'last', 'contract_id']):
    return jsonify({'error': 'Missing required fields'}), 400

  if Position.query.filter_by(contract_id=data['contract_id']).first():
    return jsonify({'error': 'Position with this contract_id already exists'}), 409

  new_position = Position(
    ticker=data['ticker'],
    strike=data.get('strike'),
    right=data.get('right'),
    last=data.get('last'),
    contract_id=data.get('contract_id'),
    iv=data.get('iv'),
    delta=data.get('delta'),
    theta=data.get('theta'),
    gamma=data.get('gamma'),
    vega=data.get('vega'),
    time_value=data.get('time_value')
  )
  db.session.add(new_position)
  db.session.commit()
  return jsonify(new_position.to_dict()), 201

# [READ] - Get all positions
@api_bp.route('/positions', methods=['GET'])
def get_all_positions():
  positions = Position.query.all()
  return jsonify([position.to_dict() for position in positions])

# [READ] - Get a single position by ID
@api_bp.route('/positions/<int:position_id>', methods=['GET'])
def get_position(position_id):
  position = db.get_or_404(Position, position_id)
  return jsonify(position.to_dict())

# [UPDATE] - Update an existing position
@api_bp.route('/positions/<int:position_id>', methods=['PUT'])
def update_position(position_id):
  position = db.get_or_404(Position, position_id)
  data = request.get_json()

  position.ticker = data.get('ticker', position.ticker)
  position.strike = data.get('strike', position.strike)
  position.right = data.get('right', position.right)
  position.last = data.get('last', position.last)
  position.contract_id = data.get('contract_id', position.contract_id)
  position.iv = data.get('iv', position.iv)
  position.delta = data.get('delta', position.delta)
  position.theta = data.get('theta', position.theta)
  position.gamma = data.get('gamma', position.gamma)
  position.vega = data.get('vega', position.vega)
  position.time_value = data.get('time_value', position.time_value)

  db.session.commit()
  return jsonify(position.to_dict())

# [DELETE] - Delete a position
@api_bp.route('/positions/<int:position_id>', methods=['DELETE'])
def delete_position(position_id):
  position = db.get_or_404(Position, position_id)
  db.session.delete(position)
  db.session.commit()
  return jsonify({'message': 'Position deleted successfully'}), 200
