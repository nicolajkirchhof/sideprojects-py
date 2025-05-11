def calculate_y_distance(line_start_point, line_end_point, points):
  """
  Calculate vertical distance between a line and points

  Args:
      line_start_point: tuple of (datetime, value) for line start
      line_end_point: tuple of (datetime, value) for line end
      points: list of tuples [(datetime, value), ...]

  Returns:
      List of vertical distances for each point
  """
  # Convert datetimes to timestamps for calculation
  x1 = line_start_point[0].timestamp()
  y1 = line_start_point[1]
  x2 = line_end_point[0].timestamp()
  y2 = line_end_point[1]

  # Calculate line slope and intercept
  slope = (y2 - y1) / (x2 - x1)
  intercept = y1 - slope * x1

  distances = []
  for point in points:
    x = point[0].timestamp()
    y = point[1]

    # Calculate y value on the line at point's x
    y_on_line = slope * x + intercept

    # Calculate vertical distance
    distance = y - y_on_line
    distances.append(distance)

  return distances
