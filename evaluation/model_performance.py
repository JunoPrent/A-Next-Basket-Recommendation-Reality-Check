from metrics import *
import pandas as pd
import json
import argparse
import os
import ast

### Start of extension
## Returns the average amount of baskets per user of a data set
def avg_dataset_baskets(data_history_dict):
    return round(np.mean([len(data_history_dict[u][1:-1]) for u in data_history_dict.keys()]))
### End of extension

def get_repeat_eval(pred_folder, dataset, size, fold_list, file):
    history_file = f'../dataset/{dataset}_history.csv'
    truth_file = f'../jsondata/{dataset}_future.json'
    with open(truth_file, 'r') as f:
        data_truth = json.load(f)
    data_history = pd.read_csv(history_file)
    a_ndcg = []
    a_recall = []
    a_hit = []
    a_repeat_ratio = []
    a_explore_ratio = []
    a_recall_repeat = []
    a_recall_explore = []
    a_hit_repeat = []
    a_hit_explore = []

    for ind in ast.literal_eval(fold_list):
        keyset_file = f'../keyset/{dataset}_keyset_{ind}.json'
        pred_file = f'{pred_folder}/{dataset}_attention_pred{ind}.json'
        with open(keyset_file, 'r') as f:
            keyset = json.load(f)
        with open(pred_file, 'r') as f:
            data_pred = json.load(f)
        # compute fold
        ndcg = []
        recall = []
        hit = []
        repeat_ratio = []
        explore_ratio = []
        recall_repeat = []
        recall_explore = []
        hit_repeat = []
        hit_explore = []


### Start of extension
        with open(f'../jsondata/{dataset}_history.json') as f:
            data_history_dict = json.load(f)
        
        threshold = avg_dataset_baskets(data_history_dict)
### End of extension

        for user in keyset['test']:
            ### Start of extension
            user_baskets = data_history_dict[user][1:-1]
            
            ## Use >= to get below average users, < for above average, comment out
            ## next two lines for all users of test set
            # if len(user_baskets) >= threshold:
            #     continue
            ### End of extension

            pred = data_pred[user]
            truth = data_truth[user][1]
            user_history = data_history[data_history['user_id'].isin([int(user)])]
            repeat_items = list(set(user_history['product_id']))
            truth_repeat = list(set(truth)&set(repeat_items)) # might be none
            truth_explore = list(set(truth)-set(truth_repeat)) # might be none

            u_ndcg = get_NDCG(truth, pred, size)
            ndcg.append(u_ndcg)
            u_recall = get_Recall(truth, pred, size)
            recall.append(u_recall)
            u_hit = get_HT(truth, pred, size)
            hit.append(u_hit)

            u_repeat_ratio, u_explore_ratio = get_repeat_explore(repeat_items, pred, size)# here repeat items
            repeat_ratio.append(u_repeat_ratio)
            explore_ratio.append(u_explore_ratio)

            if len(truth_repeat)>0:
                u_recall_repeat = get_Recall(truth_repeat, pred, size)# here repeat truth, since repeat items might not in the groundtruth
                recall_repeat.append(u_recall_repeat)
                u_hit_repeat = get_HT(truth_repeat, pred, size)
                hit_repeat.append(u_hit_repeat)

            if len(truth_explore)>0:
                u_recall_explore = get_Recall(truth_explore, pred, size)
                u_hit_explore = get_HT(truth_explore, pred, size)
                recall_explore.append(u_recall_explore)
                hit_explore.append(u_hit_explore)

        a_ndcg.append(np.mean(ndcg))
        a_recall.append(np.mean(recall))
        a_hit.append(np.mean(hit))
        a_repeat_ratio.append(np.mean(repeat_ratio))
        a_explore_ratio.append(np.mean(explore_ratio))
        a_recall_repeat.append(np.mean(recall_repeat))
        a_recall_explore.append(np.mean(recall_explore))
        a_hit_repeat.append(np.mean(hit_repeat))
        a_hit_explore.append(np.mean(hit_explore))
        if ind == 0:
            print(f'fold_id,    recall,    ndcg,    phr')
            file.write(f'fold_id,    recall,    ndcg,    phr\n')
        print(f'{ind},    {np.mean(recall):.10f},    {np.mean(ndcg): .10f},    {np.mean(hit):.10f}')
        file.write(f'{ind},    {np.mean(recall):.10f},    {np.mean(ndcg): .10f},    {np.mean(hit):.10f}\n')
    print('average over folds:')
    print('basket size:', size)
    print('recall, ndcg, phr:', np.mean(a_recall), np.mean(a_ndcg), np.mean(a_hit))
    print('repeat-explore ratio:', np.mean(a_repeat_ratio), np.mean(a_explore_ratio))
    print('repeat-explore recall', np.mean(a_recall_repeat), np.mean(a_recall_explore))
    print('repeat-explore phr:', np.mean(a_hit_repeat), np.mean(a_hit_explore))

    file.write('basket size: ' + str(size) + '\n')
    file.write('recall, ndcg, phr: '+ str(np.mean(a_recall)) +' ' +str(np.mean(a_ndcg))+' '+ str(np.mean(a_hit)) +'\n')
    file.write('repeat-explore ratio:'+ str(np.mean(a_repeat_ratio)) +' ' +str(np.mean(a_explore_ratio)) +'\n')
    file.write('repeat-explore recall' + str(np.mean(a_recall_repeat)) + ' ' + str(np.mean(a_recall_explore)) +'\n')
    file.write('repeat-explore phr:' + str(np.mean(a_hit_repeat)) + ' ' + str(np.mean(a_hit_explore)) + '\n')
    return np.mean(a_recall)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=str, required=True, help='x')
    parser.add_argument('--fold_list', type=str, required=True, help='x')
    args = parser.parse_args()
    pred_folder = args.pred_folder
    fold_list = args.fold_list
    eval_file = 'eval_results.txt'
    f = open(eval_file, 'w')
    for dataset in ['dunnhumby', 'tafeng', 'instacart']:
        f.write('############'+dataset+'########### \n')
        print(dataset)
        get_repeat_eval(pred_folder, dataset, 10, fold_list, f)
        get_repeat_eval(pred_folder, dataset, 20, fold_list, f)
