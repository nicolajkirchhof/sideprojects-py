
import pytz
from datetime import datetime, timedelta
from dateutil import parser
#%%

def get_last_saturday(date):
  """
  Returns the last Saturday before or equal to the given date.

  Args:
      date (datetime): The arbitrary input date.

  Returns:
      datetime: The last Saturday.
  """
  # Saturday is represented by 5 in Python's `weekday()` function (where Monday is 0 and Sunday is 6)
  days_to_saturday = (date.weekday() - 5) % 7
  # Subtract the appropriate number of days to get to Saturday
  last_saturday = date - timedelta(days=days_to_saturday)
  return last_saturday
