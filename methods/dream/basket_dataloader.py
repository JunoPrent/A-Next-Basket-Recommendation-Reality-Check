
from torch.utils.data import Dataset, DataLoader
import json
from candidates import *

### Start of extension
## Returns the average amount of baskets per user of a data set
def avg_dataset_baskets(data_history_dict):
    return round(np.mean([len(data_history_dict[u][1:-1]) for u in data_history_dict.keys()]))
### End of extension

class BasketDataset(Dataset):

    def __init__(self, config, mode='train'):
        # load data.

        train_file = config['train_file']
        tgt_file = config['tgt_file']
        data_config_file = config['data_config_file']
        with open(train_file, 'r') as f:
            data_all = json.load(f)
        with open(tgt_file, 'r') as f:
            tgt_all = json.load(f)
        with open(data_config_file, 'r') as f:
            data_config = json.load(f)

        self.total_num = data_config['item_num'] # item total num
        self.mode = mode
        # Could do some filtering here
        self.data = []
        self.tgt = []
        
        ### Start of extension
        ## Use this to get users with at least the average amount of baskets
        threshold = avg_dataset_baskets(data_all)

        ## Or use this to get the top 20% most frequently ordering users
        ## Use value 13 for Dunnhumby, 8 for TaFeng, 20 for Instacart
        # threshold = 13
        ### End of extension

        for user in data_config[mode]:
            ### Start of extension
            # if len(data_all[user][1:-1]) <= threshold and mode in ['train', 'val']:
            #     continue
            ### End of extension

            self.data.append(data_all[user][1:-1])
            self.tgt.append(tgt_all[user][1])

        self.user_sum = len(self.data)
        self.repeat_candidates = get_repeat_candidates(self.data)
        self.explore_candidates = get_explore_candidates(self.data, self.total_num)

    def __getitem__(self, ind):
        return self.data[ind], self.tgt[ind], \
               self.repeat_candidates[ind], self.explore_candidates[ind]

    def get_batch_data(self, s, e):
        return self.data[s:e], self.tgt[s:e], \
               self.repeat_candidates[s:e], self.explore_candidates[s:e]

    def __len__(self):
        return self.user_sum

def load_datafile(train_file, tgt_file, data_config_file):

    # load data.
    with open(train_file, 'r') as f:
        data = json.load(f)
    with open(tgt_file, 'r') as f:
        tgt = json.load(f)
    with open(data_config_file, 'r') as f:
        data_config = json.load(f)

    # Could do some filtering here
    data_train = []
    tgt_train = []
    for user in data_config['train']:
        data_train.append(data[user])
        tgt_train.append(tgt[user])
    # Valid
    data_val = []
    tgt_val = []
    for user in data_config['meta']:
        data_val.append(data[user])
        tgt_val.append(tgt[user])
    #test
    data_test = []
    tgt_test = []
    for user in data_config['test']:
        data_test.append(data[user])
        tgt_test.append(tgt[user])

    return data_train, tgt_train, data_val, tgt_val, data_test, tgt_test

if __name__ =='__main__':
    train_file = '../jsondata/Instacart_baskets_train.json'
    tgt_file = '../jsondata/Instacart_baskets_test.json'
    config_file = '../jsondata/Instacart_keyset.json'
    dataset_s = BasketDataset(train_file, tgt_file, config_file)
    print(dataset_s.get_batch_data(0,4))