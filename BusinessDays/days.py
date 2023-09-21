#%%
from datetime import datetime, timedelta
import locale
#%%
# initializing dates ranges
date_ranges = [[datetime(2023, 10, 2),datetime(2023, 10, 15)],[datetime(2023, 10, 23), datetime(2023, 11, 18)]]


dates = [d for d in [d1 + timedelta(idx) for [d1, d2] in date_ranges for idx in range((d2 - d1).days)] if d.weekday() < 5]



locale.setlocale(locale.LC_ALL, "de_DE")
for date in dates:
    print(date.strftime("%A, %x"))
