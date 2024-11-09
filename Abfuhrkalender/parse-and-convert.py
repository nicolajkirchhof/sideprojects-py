#%%
import pprint

import ics
import warnings
warnings.filterwarnings("ignore")

#%%
with open('Abfuhrkalender/Abfuhrkalender_Hildesheim.ics', encoding='utf-8') as f:
  c = ics.Calendar(f.read())
#%%
events_no_biomuell = {e for e in c.events if e.name.find('Bio') < 0}
events_biomuell = {e for e in c.events if e.name.find('Bio') >= 0}
pprint.pprint([e.name for e in events_no_biomuell])

#%%
c_export = c.clone()
c_export.events = events_no_biomuell
[e.name for e in c_export.events]
#%%
with open('Cleaned_Abfuhrkalender.ics', 'w', encoding='utf-8') as f:
  f.writelines(c_export.serialize_iter())
