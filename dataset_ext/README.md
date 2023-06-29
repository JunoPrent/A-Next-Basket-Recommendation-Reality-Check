Hereby you can find the information how to train and test DREAM on different dataset sizes, namely the sizes: 22000, 11000, 5500, 2250 and 1125. Where each next dataset size is a decrease of the previous one by 50%.


## Adjustments and pre-processing
* dunnhumbyconf.json file: Depending on where you are running the code, the '../' parts before the file imports might have to be removed
* Run the pre-processing file to create the datasets

## Training the model
For each of the different datasets:
* In the dunnhumbyconf.jsn file: 
- adjust the "train_file": "jsondata/dunnhumby_history.json" to /dataset_ext/history_sub{your_number}.json
- adjust the "tgt_file": "jsondata/dunnhumby_future.json" to /dataset_ext/future_sub{your_number}.json
* Use command: python3 methods/dream/trainer.py --dataset dunnhumby --fold_id 0 --attention 1   to train the model 
### NOTE: HERE AN ERROR IS PRODUCED CAUSED BY AN ERROR OCCURING IN THE BASKET_DATALOADER.PY FILE. IN LINE 50, DATA_ALL[USER] CONTAINS USERS WHICH ARE NOT IN THE CREATED SUBSET OF THE ORIGINAL DATASET. THE SAME ERROR IS OCCURING FOR TGT_ALL[USER]. TO BE RESOLVED.
* Use command: python3 methods/dream/pred_results.py --dataset dunnhumby --fold_id 0 to test the model 
* See results
