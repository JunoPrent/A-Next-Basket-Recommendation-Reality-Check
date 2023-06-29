import pandas as pd
from tqdm import tqdm
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_item_order', action='store_true')
    args = parser.parse_args()
    use_item_order = args.use_item_order

    user_order_d = pd.read_csv('../DataSource/instacart/orders.csv',
                            usecols=['user_id', 'order_number', 'order_id', 'eval_set'])
    # also load the 'add_to_cart_order' column
    order_item_train = pd.read_csv('../DataSource/instacart/order_products__train.csv',
                                usecols=['order_id', 'product_id', 'add_to_cart_order'])
    order_item_prior = pd.read_csv('../DataSource/instacart/order_products__prior.csv',
                                usecols=['order_id', 'product_id', 'add_to_cart_order'])
    order_item = pd.concat([order_item_prior, order_item_train], ignore_index=True)
    user_order = pd.merge(user_order_d, order_item, on='order_id', how='left')
    user_order = user_order.dropna(how='any')

    # convert 'add_to_cart_order' to datatype integer
    user_order['add_to_cart_order'] = user_order['add_to_cart_order'].astype(int)

    # take subset (10%) of all users in dataset 
    user_num = len(set(user_order['user_id'].tolist()))
    user_num = int(user_num*0.1)
    user_order = user_order[user_order['user_id'] <= user_num]

    # create rows of dataframe
    ##### optimized code #####
    grouped_users = user_order.groupby('user_id')
    baskets_dict = {'user_id': [], 'order_number': [], 'product_id': [], 'add_to_cart_order': [], 'eval_set': []}
    for user, user_data in tqdm(grouped_users):
        order_list = list(set(user_data['order_number'].tolist()))
        order_list = sorted(order_list)
        # only consider users with 3 to 50 baskets (orders)
        if len(order_list)>=3 and len(order_list)<=50:
            basket_idx = 1
            for orig_basket_idx in order_list:
                # select rows of order
                basket_data = user_data[user_data['order_number'].isin([orig_basket_idx])]
                basket_items = list(basket_data['product_id'].tolist())
                item_num = len(basket_items)
                # store values in dict which will become the rows of a dataframe
                baskets_dict['user_id'].extend([user for _ in range(item_num)])
                baskets_dict['order_number'].extend([basket_idx for _ in range(item_num)])          # order_number is converted to per users
                baskets_dict['product_id'].extend(basket_items)
                baskets_dict['add_to_cart_order'].extend(basket_data['add_to_cart_order'].tolist())
                # add the last basket of a user to the train instead of prior
                if orig_basket_idx == order_list[-1]:
                    baskets_dict['eval_set'].extend(['train' for _ in range(item_num)])
                else:
                    baskets_dict['eval_set'].extend(['prior' for _ in range(item_num)])
                basket_idx += 1
    baskets = pd.DataFrame(data=baskets_dict)
    ##### old implementation #####
    # baskets = None
    # for user, user_data in tqdm(user_order.groupby('user_id')):
    #     date_list = list(set(user_data['order_number'].tolist()))
    #     date_list = sorted(date_list)
    #     # print(date_list)
    #     if len(date_list)>=3 and len(date_list)<=50:
    #         date_num = 1
    #         for date in date_list:
    #             date_data = user_data[user_data['order_number'].isin([date])]
    #             date_item = list(set(date_data['product_id'].tolist()))
    #             item_num = len(date_item)
    #             # concat slows down when dfs get larger
    #             if baskets is None:
    #                 baskets = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
    #                                         'order_number': pd.Series([date_num for i in range(item_num)]),
    #                                         'product_id': pd.Series(date_item),
    #                                         'eval_set': pd.Series(['prior' for i in range(item_num)])})
    #                 date_num += 1
    #             else:
    #                 if date == date_list[-1]:#if date is the last. then add a tag here
    #                     temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
    #                                             'order_number': pd.Series([date_num for i in range(item_num)]),
    #                                             'product_id': pd.Series(date_item),
    #                                             'eval_set': pd.Series(['train' for i in range(item_num)])})
    #                     date_num += 1
    #                     baskets = pd.concat([baskets, temp], ignore_index=True)
    #                 else:
    #                     temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
    #                                             'order_number': pd.Series([date_num for i in range(item_num)]),
    #                                             'product_id': pd.Series(date_item),
    #                                             'eval_set': pd.Series(['prior' for i in range(item_num)])})
    #                     date_num += 1
    #                     baskets = pd.concat([baskets, temp], ignore_index=True)
    print('total transcations:', len(baskets))

    ##### old code, only added comments #####
    # filter out items from the baskets based on their frequency in the 'prior' data
    item_set_all = set()
    item_filter_dict = dict()
    history_baskets = baskets[baskets['eval_set'].isin(['prior'])].reset_index()
    # first count items
    for ind in range(len(history_baskets)):
        product_id = history_baskets['product_id'].iloc[ind]
        if product_id not in item_filter_dict:
            item_filter_dict[product_id] = 1
        else:
            item_filter_dict[product_id] += 1
    # select items with sufficient frequency
    for key in item_filter_dict.keys():
        if item_filter_dict[key]>=17:
            item_set_all.add(key)
    # filter out low frequency items from baskets
    print('Filter data use the training items.')
    baskets = baskets[baskets['product_id'].isin(item_set_all)].reset_index()
    print('After transcations:', len(baskets))


    # reindex user and product ids
    ##### optimized code #####
    baskets['product_id'] = pd.factorize(baskets['product_id'])[0]          # note that the product factorization should start at 0 instead of 1
    baskets['user_id'] = pd.factorize(baskets['user_id'])[0] + 1
    if use_item_order:
        baskets = baskets.loc[:, ['user_id', 'order_number', 'product_id', 'add_to_cart_order', 'eval_set']]
    else: 
        baskets = baskets.loc[:, ['user_id', 'order_number', 'product_id', 'eval_set']]
    ##### old code #####
    # item_dict = dict()
    # item_ind = 1
    # user_dict = dict()
    # user_ind = 1
    # for ind in range(len(baskets)):
    #     product_id = baskets.at[ind, 'product_id']
    #     if product_id not in item_dict:
    #         item_dict[product_id] = item_ind
    #         item_ind += 1
    #     baskets.at[ind, 'product_id'] = item_dict[product_id]

    #     user_id = baskets.at[ind, 'user_id']
    #     if user_id not in user_dict:
    #         user_dict[user_id] = user_ind
    #         user_ind += 1
    #     baskets.at[ind, 'user_id'] = user_dict[user_id]
    # baskets = baskets.loc[:, ['user_id', 'order_number', 'product_id', 'add_to_cart_order', 'eval_set']]

    ##### old code #####
    # baskets.to_csv('../dataset/instacart.csv', index=False)
    # directly write to two csv files for prior and train
    ##### new code #####
    df_fut = baskets.loc[baskets['eval_set'] == 'train'].drop(columns='eval_set')
    df_hist = baskets.loc[baskets['eval_set'] == 'prior'].drop(columns='eval_set')
    if use_item_order:
        file_str_part = '_itemorder'
    else:
        file_str_part = ''
    print('Writing to csv...')
    df_fut.to_csv(f'../dataset/instacart{file_str_part}_future.csv', index=False)
    df_hist.to_csv(f'../dataset/instacart{file_str_part}_history.csv', index=False)


    ##### convert to the json format #####
    '''
    No code for this conversion was provided, therefore this is done manually by inspecting the format of the json files.
    '''
    # loop over both the future and history dataframes
    print('Writing to json...')
    for df, df_name in zip([df_fut, df_hist], ['future', 'history']):
        # create the baskets for each user per order (create baskets)
        prods_grouped = df.groupby(['user_id', 'order_number'])['product_id'].agg(list)
        # get rid of order_number from index
        prods_grouped = prods_grouped.reset_index(level='order_number')
        # combine the different baskets for each user in a nested list
        prods_grouped = prods_grouped.groupby('user_id')['product_id'].agg(list)
        # encode following the template style of original json files
        data_dict_products = {}
        # loop over users, products is a nested list with the baskets
        for user_id, user_baskets in prods_grouped.items():
            # user_id as key of dict followed by a list of the format [[-1], [basket_1], [basket_2] .... [basket_n], [-1]] so unpack the user_baskets
            data_dict_products[str(user_id)] = [[-1], *user_baskets, [-1]]
        # store dictionary as json format
        json_data_prods = json.dumps(data_dict_products)
        with open(f'../jsondata/instacart{file_str_part}_'+df_name+'.json', 'w') as f:
            f.write(json_data_prods)

        # if item order is used, create separate json file for storing the add to cart order in the same format as the baskets per user
        if use_item_order:
            addorder_grouped = df.groupby(['user_id', 'order_number'])['add_to_cart_order'].agg(list)
            addorder_grouped = addorder_grouped.reset_index(level='order_number')
            addorder_grouped = addorder_grouped.groupby('user_id')['add_to_cart_order'].agg(list)
            data_dict_addorder = {}
            for user_id, user_addorders in addorder_grouped.items():
                data_dict_addorder[str(user_id)] = [[-1], *user_addorders, [-1]]
            json_data_addorders = json.dumps(data_dict_addorder)
            with open(f'../jsondata/instacart{file_str_part}_orders_'+df_name+'.json', 'w') as f:
                f.write(json_data_addorders)
    print('Data preprocessing complete')