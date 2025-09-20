from .extensions import db

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


class TradeLog(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  trade_id = db.Column(db.Integer, db.ForeignKey('trade.id'), nullable=False)
  date = db.Column(db.DateTime, nullable=False)
  rich_text_notes = db.Column(db.Text, nullable=True)

  def to_dict(self):
    return {
      'id': self.id,
      'trade_id': self.trade_id,
      'date': self.date.isoformat(),
      'rich_text_notes': self.rich_text_notes
    }

  def __repr__(self):
    return f'<TradeLog {self.id}>'

class TradeIdea(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  date = db.Column(db.DateTime, nullable=False)
  rich_text_notes = db.Column(db.Text, nullable=True)

  def to_dict(self):
    return {
      'id': self.id,
      'date': self.date.isoformat(),
      'rich_text_notes': self.rich_text_notes
    }

  def __repr__(self):
    return f'<TradeIdea {self.id}>'

class Position(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  trade_id = db.Column(db.Integer, db.ForeignKey('trade.id'), nullable=False)
  contract_id = db.Column(db.String(20), nullable=False)
  type = db.Column(db.String(50), nullable=False)
  opened = db.Column(db.DateTime, nullable=False)
  expiry = db.Column(db.DateTime, nullable=False)
  closed = db.Column(db.DateTime, nullable=True)
  position = db.Column(db.Integer, nullable=False)
  right = db.Column(db.String(1), nullable=True)
  strike = db.Column(db.Float, nullable=False)
  cost = db.Column(db.Float, nullable=False)
  close = db.Column(db.Float, nullable=True)
  multiplier = db.Column(db.Integer, nullable=False)

  trade = db.relationship('Trade', backref=db.backref('positions', lazy=True))

  def to_dict(self):
    return {
      'id': self.id,
      'trade_id': self.trade_id,
      'contract_id': self.contract_id,
      'type': self.type,
      'opened': self.opened.isoformat(),
      'expiry': self.expiry.isoformat(),
      'closed': self.closed.isoformat() if self.closed else None,
      'position': self.position,
      'right': self.right,
      'strike': self.strike,
      'cost': self.cost,
      'close': self.close,
      'multiplier': self.multiplier
    }

  def __repr__(self):
    return f'<Position {self.id}>'
