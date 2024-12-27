import numpy as np

def percentage_change(old_value, new_value):
  """Calculates the percentage change between two numbers.

    Args:
        old_value: The original value.
        new_value: The new value.

    Returns:
        The percentage change, or NaN if the old value is zero.
    """
  if old_value == 0:
    return float('nan')  # Or handle it differently, e.g., return 0 or raise an exception
  return ((new_value - old_value) / old_value) * 100


def subtract_percentage(number, percentage):
  """Subtracts a percentage from a number.

    Args:
        number: The number to subtract from.
        percentage: The percentage to subtract (as a decimal or integer).

    Returns:
        The result of subtracting the percentage from the number.
    """
  pct = percentage / 100.0
  amount_to_subtract = number * pct
  result = number - amount_to_subtract
  return result

def add_percentage(number, percentage):
  """Subtracts a percentage from a number.

    Args:
        number: The number to subtract from.
        percentage: The percentage to subtract (as a decimal or integer).

    Returns:
        The result of subtracting the percentage from the number.
    """
  pct = percentage / 100.0
  amount_to_subtract = number * pct
  result = number + amount_to_subtract
  return result

def profit_loss(position, close_price):
  return position.size * (close_price - position.price)

def angle_rowwise_v2(A, B):
  p1 = np.einsum('ij,ij->i',A,B)
  p2 = np.einsum('ij,ij->i',A,A)
  p3 = np.einsum('ij,ij->i',B,B)
  p4 = p1 / np.sqrt(p2*p3)
  return np.arccos(np.clip(p4,-1.0,1.0))*180/np.pi
