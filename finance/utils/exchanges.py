from datetime import timedelta

from zoneinfo import ZoneInfo

DE_EXCHANGE = {'Region': 'EU', 'TZ':ZoneInfo('Europe/Berlin'), 'Open': timedelta(hours=9), 'Close': timedelta(hours=17, minutes=30), 'PostClose': timedelta(hours=22), 'PreOpen': timedelta(hours=8)}
GB_EXCHANGE = {'Region': 'EU', 'TZ':ZoneInfo('Europe/London'), 'Open': timedelta(hours=8), 'Close': timedelta(hours=16, minutes=30), 'PostClose': timedelta(hours=17), 'PreOpen': timedelta(hours=4, minutes=30)}
US_EXCHANGE = {'Region': 'US', 'TZ':ZoneInfo('America/Chicago'), 'Open': timedelta(hours=8, minutes=30), 'Close': timedelta(hours=16), 'PostClose': timedelta(hours=17), 'PreOpen': timedelta(hours=4, minutes=30)}
JP_EXCHANGE = {'Region': 'JP', 'TZ':ZoneInfo('Asia/Tokyo'), 'Open': timedelta(hours=8, minutes=45), 'Close': timedelta(hours=15, minutes=45), 'PostClose': timedelta(hours=30), 'PreOpen': timedelta(hours=8, minutes=45)}
HK_EXCHANGE = {'Region': 'HK', 'TZ':ZoneInfo('Asia/Hong_Kong'), 'Open': timedelta(hours=9, minutes=30), 'Close': timedelta(hours=16, minutes=0), 'PostClose': timedelta(hours=26), 'PreOpen': timedelta(hours=9, minutes=30)}
AU_EXCHANGE = {'Region': 'AU', 'TZ':ZoneInfo('Australia/Sydney'), 'Open': timedelta(hours=10, minutes=0), 'Close': timedelta(hours=16, minutes=0), 'PostClose': timedelta(hours=19), 'PreOpen': timedelta(hours=7, minutes=0)}
US_NY_EXCHANGE = {'Region': 'US', 'TZ':ZoneInfo('America/New_York'), 'Open': timedelta(hours=0), 'Close': timedelta(hours=24), 'PostClose': timedelta(hours=24), 'PreOpen': timedelta(hours=0)}
