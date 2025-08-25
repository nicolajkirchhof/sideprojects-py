from app import create_app
import os

app = create_app()

if __name__ == '__main__':
  # Make sure to set FLASK_APP environment variable for flask commands
  os.environ['FLASK_APP'] = 'run.py'
  app.run(debug=True)
