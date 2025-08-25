import os
from dotenv import load_dotenv

load_dotenv()

class Config:
  SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                            'postgresql://nico:F70247356A6CE05946B379BB8435C015@localhost/'
  SQLALCHEMY_TRACK_MODIFICATIONS = False
