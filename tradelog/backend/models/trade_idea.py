from ..extensions import db

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
