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
    conf['loss_mode'] = 0  # bceloss
    para_path = glob.glob(pathname='models/'+dataset.split('_')[0]+'/*')
    keyset_file = '../keyset/'+dataset+'_keyset_'+str(fold_id)+'.json'
    print(keyset_file)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    conf['item_num'] = keyset['item_num']
    conf['device'] = torch.device("cpu")
    keyset_test = keyset['test']

    checkpoint_file = []
    # if no checkpoint path provided: use most recent file for the specified dataset and fold_id
    if not cp_path:
        for path in para_path:
            path_l = path.split('-')
            if path_l[0].split('\\')[-1] == dataset and path_l[3] == str(fold_id):
                checkpoint_file.append(path)
    else:
        checkpoint_file.append(cp_path)
    checkpoint = torch.load(checkpoint_file[0], map_location=torch.device('cpu'))
    ##### new code ######
    # check for usage of itemorder embedding layer in checkpoint state dict
    if 'basket_embedding.addorder_embedding.weight' in checkpoint['state_dict'].keys():
        print('present')
        conf['use_item_order'] = True
        conf['max_addorder'] = keyset['max_addorder']
        use_item_order_str = 'itemorder_'
        # load addorder file
        history_addorder_file = '../jsondata/'+dataset+'_orders_history.json'
        with open(history_addorder_file, 'r') as f:
            data_history_addorder = json.load(f)
    else: 
        conf['use_item_order'] = False
        use_item_order_str = ''
    # check for usage of attention in checkpoint model
    if 'decoder.W_repeat.weight' in checkpoint['state_dict'].keys():
        conf['attention'] = True
        use_attention_str = 'attention_'
    else: 
        conf['attention'] = False
        use_attention_str = ''
    # create separte folders for the predictions per dataset used
    pred_file = f'pred/{dataset}_{use_attention_str}{use_item_order_str}pred{str(fold_id)}.json'
    # initialize model with configuration and keysets
    model = NBRNet(conf, keyset)
    model.load_state_dict(checkpoint['state_dict'])
    ######################
    message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
    print(message_output)
    model.eval()
    pred_dict = dict()
    for user in keyset_test:
        basket = [data_history[user][1:-1]]
        cand = [[item for item in range(keyset['item_num'])]]
        if conf['use_item_order']:
            addorder = [data_history_addorder[user][1:-1]]
        else:
            addorder = None
        scores = model.forward(basket, addorder, cand)
        pred_dict[user] = scores[0].detach().numpy().argsort()[::-1][:100].tolist()

    with open(pred_file, 'w') as f:
        json.dump(pred_dict, f)
