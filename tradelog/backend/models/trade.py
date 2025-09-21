from ..extensions import db

class Trade(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  symbol = db.Column(db.String(50), nullable=False)
  profit_mechanism = db.Column(db.String(100), nullable=False)

  def to_dict(self):
    return {
      'id': self.id,
      'symbol': self.symbol,
      'profit_mechanism': self.profit_mechanism
    }

  def __repr__(self):
    return f'<Trade {self.id}>'
