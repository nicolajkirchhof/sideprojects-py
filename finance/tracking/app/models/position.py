from app import db
from datetime import date
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Date, Float, Integer

class Position(db.Model):
  id: Mapped[int] = mapped_column(Integer, primary_key=True)
  date: Mapped[date] = mapped_column(Date, nullable=False, default=date.today)
  ticker: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
  strike: Mapped[float] = mapped_column(Float, nullable=False)
  right: Mapped[str] = mapped_column(String(4), nullable=False)  # 'CALL' or 'PUT'
  last: Mapped[float] = mapped_column(Float, nullable=False)
  contract_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
  iv: Mapped[float] = mapped_column(Float, nullable=True)
  delta: Mapped[float] = mapped_column(Float, nullable=True)
  theta: Mapped[float] = mapped_column(Float, nullable=True)
  gamma: Mapped[float] = mapped_column(Float, nullable=True)
  vega: Mapped[float] = mapped_column(Float, nullable=True)
  time_value: Mapped[float] = mapped_column(Float, nullable=True)

  def to_dict(self):
    """Converts the Position object to a dictionary."""
    return {
      'id': self.id,
      'date': self.date.isoformat(),
      'ticker': self.ticker,
      'strike': self.strike,
      'right': self.right,
      'last': self.last,
      'contract_id': self.contract_id,
      'iv': self.iv,
      'delta': self.delta,
      'theta': self.theta,
      'gamma': self.gamma,
      'vega': self.vega,
      'time_value': self.time_value
    }

  def __repr__(self):
    return f'<Position {self.contract_id}>'
