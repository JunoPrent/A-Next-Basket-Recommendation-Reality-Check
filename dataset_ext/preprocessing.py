
### CREATING SUBSETS OF THE DUNNHUMBY DATASET FOR FUTURE AND HISTORY

### IMPROTING MODULES
from modules import *

### FUTURE
# Opening JSON file
f = open('jsondata/dunnhumby_future.json')
data = json.load(f) # returns JSON object as a dictionary
f.close()

future_subs1 = dict(list(data.items())[:1124])
future_subs2 = dict(list(data.items())[:2249])
future_subs3 = dict(list(data.items())[:4499])
future_subs4 = dict(list(data.items())[:10999])
future_subs5 = dict(list(data.items())[:21999])
future_subsets = [future_subs1,future_subs2,future_subs3,future_subs4,future_subs5]
future_subset_names = ['future_subs1','future_subs2','future_subs3','future_subs4','future_subs5']

for i, subset in enumerate(future_subset_names):
    filename = str(subset)+'.json'
    print(filename)
    with open('dataset_ext/'+str(filename), 'w') as fp:
        json.dump(future_subsets[i], fp)
    fp.close()


### HISTORY
# Opening JSON file
f = open('jsondata/dunnhumby_history.json')
data = json.load(f) # returns JSON object as a dictionary
f.close()

history_subs1 = dict(list(data.items())[:1124])
history_subs2 = dict(list(data.items())[:2249])
history_subs3 = dict(list(data.items())[:4499])
history_subs4 = dict(list(data.items())[:10999])
history_subs5 = dict(list(data.items())[:21999])
history_subsets = [history_subs1,history_subs2,history_subs3,history_subs4,history_subs5]
history_subset_names = ['history_subs1','history_subs2','history_subs3','history_subs4','history_subs5']

for i, subset in enumerate(history_subset_names):
    filename = str(subset)+'.json'
    print(filename)
    with open('dataset_ext/'+str(filename), 'w') as fp:
        json.dump(history_subsets[i], fp)
    fp.close()