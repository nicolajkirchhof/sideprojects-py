'''
This script refreshes sensor data for the Swing PM project.
It fetches data from the sensor API and updates the local database.
'''

from finance import utils

#%%

for symbol in utils.underlyings.get_liquid_underlyings():
  utils.
