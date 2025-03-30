from datetime import timedelta

import pytz

DE_EXCHANGE = {'TZ':pytz.timezone('Europe/Berlin'), 'Open': timedelta(hours=9), 'Close': timedelta(hours=17, minutes=30), 'PostClose': timedelta(hours=22), 'PreOpen': timedelta(hours=8)}
GB_EXCHANGE = {'TZ':pytz.timezone('Europe/London'), 'Open': timedelta(hours=8), 'Close': timedelta(hours=16, minutes=30), 'PostClose': timedelta(hours=17), 'PreOpen': timedelta(hours=4, minutes=30)}
US_EXCHANGE = {'TZ':pytz.timezone('America/Chicago'), 'Open': timedelta(hours=8, minutes=30), 'Close': timedelta(hours=16), 'PostClose': timedelta(hours=17), 'PreOpen': timedelta(hours=4, minutes=30)}
JP_EXCHANGE = {'TZ':pytz.timezone('Asia/Tokyo'), 'Open': timedelta(hours=8, minutes=45), 'Close': timedelta(hours=15, minutes=45), 'PostClose': timedelta(hours=30), 'PreOpen': timedelta(hours=8, minutes=45)}
HK_EXCHANGE = {'TZ':pytz.timezone('Asia/Hong_Kong'), 'Open': timedelta(hours=9, minutes=30), 'Close': timedelta(hours=16, minutes=0), 'PostClose': timedelta(hours=26), 'PreOpen': timedelta(hours=9, minutes=30)}
AU_EXCHANGE = {'TZ':pytz.timezone('Australia/Sydney'), 'Open': timedelta(hours=10, minutes=0), 'Close': timedelta(hours=16, minutes=0), 'PostClose': timedelta(hours=19), 'PreOpen': timedelta(hours=7, minutes=0)}
US_NY_EXCHANGE = {'TZ':pytz.timezone('America/New_York'), 'Open': timedelta(hours=0), 'Close': timedelta(hours=24), 'PostClose': timedelta(hours=24), 'PreOpen': timedelta(hours=0)}
