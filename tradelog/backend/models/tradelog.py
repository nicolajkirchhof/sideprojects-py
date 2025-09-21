from ..extensions import db

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
