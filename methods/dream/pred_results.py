import json
import glob
from Explainablebasket import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to saved checkpoint')
    args = parser.parse_args()
    dataset = args.dataset
    fold_id = args.fold_id
    cp_path = args.checkpoint_path
    history_file = '../jsondata/'+dataset+'_history.json'
    future_file = '../jsondata/'+dataset+'_future.json'
    with open(history_file, 'r') as f:
        data_history = json.load(f)
    with open(future_file, 'r') as f:
        data_future = json.load(f)
    with open('dream/' + dataset+'conf.json', 'r') as f:
        conf = json.load(f)
    for mode in ['attention']:
        conf['attention'] = mode
        conf['loss_mode'] = 0  # bceloss
        para_path = glob.glob(pathname='models/'+dataset+'/*')
        keyset_file = '../keyset/'+dataset+'_keyset_'+str(fold_id)+'.json'
        print(keyset_file)
        pred_file = 'pred/'+dataset+'_'+mode+'_pred'+str(fold_id)+'.json'
        with open(keyset_file, 'r') as f:
            keyset = json.load(f)
        conf['item_num'] = keyset['item_num']
        conf['device'] = torch.device("cpu")
        keyset_test = keyset['test']

        checkpoint_file = []
        if not cp_path:
            for path in para_path:
                path_l = path.split('-')
                if path_l[4] == mode and path_l[3] == str(fold_id):
                    checkpoint_file.append(path)
        else:
            checkpoint_file.append(cp_path)
        model = NBRNet(conf, keyset)
        checkpoint = torch.load(checkpoint_file[0], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
        print(message_output)
        model.eval()
        pred_dict = dict()
        for user in keyset_test:
            basket = [data_history[user][1:-1]]
            cand = [[item for item in range(keyset['item_num'])]]
            scores = model.forward(basket, cand)
            pred_dict[user] = scores[0].detach().numpy().argsort()[::-1][:100].tolist()

        with open(pred_file, 'w') as f:
            json.dump(pred_dict, f)
