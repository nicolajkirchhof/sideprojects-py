import ib_async as ib

eu_indices = [ib.Index(x, 'EUREX', 'EUR') for x in ['DAX', 'ESTX50']]
us_indices = [ib.Index('SPX', 'CBOE', 'USD'), ib.Index('NDX', 'NASDAQ', 'USD'),
              ib.Index('RUT', 'RUSSELL', 'USD'), ib.Index('INDU', 'CME', 'USD')]
fr_index = ib.Index('CAC40', 'MONEP', 'EUR')
es_index = ib.Index('IBEX35', 'MEFFRV', 'EUR')
gb_index = ib.CFD('IBGB100', 'SMART', 'GBP')
jp_index = ib.Index('N225', 'OSE.JPN', 'JPY')
aus_index = ib.CFD('IBAU200', 'SMART', 'AUD')
hk_index = ib.Index('HSI', 'HKFE', 'HKD')
