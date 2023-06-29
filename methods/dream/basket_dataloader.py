
from torch.utils.data import Dataset, DataLoader
import json
from candidates import *
import os

class BasketDataset(Dataset):

    def __init__(self, config, mode='train'):
        # load data.

        train_file = config['train_file']
        train_addorder_file = config['train_addorder_file']
        tgt_file = config['tgt_file']
        tgt_addorder_file = config['tgt_addorder_file']
        data_config_file = config['data_config_file']
        with open(train_file, 'r') as f:
            data_all = json.load(f)
        with open(train_addorder_file, 'r') as f:
            train_addorder_all = json.load(f)
        with open(tgt_file, 'r') as f:
            tgt_all = json.load(f)
        with open(tgt_addorder_file, 'r') as f:
            tgt_addorder_all = json.load(f)
        with open(data_config_file, 'r') as f:
            data_config = json.load(f)

        self.total_num = data_config['item_num'] # item total num
        self.mode = mode
        # Could do some filtering here
        self.data = []
        self.data_addorder = []
        self.tgt = []
        self.tgt_addorder = []
        for user in data_config[mode]:
            self.data.append(data_all[user][1:-1])
            self.data_addorder.append(train_addorder_all[user][1:-1])
            # use index 1 since future files only contain 1 basket per user
            self.tgt.append(tgt_all[user][1])
            self.tgt_addorder.append(tgt_addorder_all[user][1])
        self.user_sum = len(self.data)
        self.repeat_candidates = get_repeat_candidates(self.data)
        # optimized candidates funciton in candidates.py
        self.explore_candidates = get_explore_candidates(self.data, self.total_num)

    def __getitem__(self, ind):
        return self.data[ind], self.tgt[ind], \
               self.repeat_candidates[ind], self.explore_candidates[ind], \
               self.data_addorder[ind], self.tgt_addorder[ind]

    def get_batch_data(self, s, e):
        return self.data[s:e], self.tgt[s:e], \
               self.repeat_candidates[s:e], self.explore_candidates[s:e], \
               self.data_addorder[s:e], self.tgt_addorder[s:e]

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
    # cd methods
    with open('./dream/instacartconf.json', 'r') as f:
        train_config = json.load(f)
    train_config['data_config_file'] = "../keyset/instacart_keyset_0.json"
    basket_dataset = BasketDataset(train_config, mode='train')
    validate_dataset = BasketDataset(train_config, mode='val')
    # print(dataset_s.get_batch_data(0,1)[-1])