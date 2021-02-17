import os
from os.path import dirname as dr, abspath
import csv
import pickle
import random
import statistics
import torch
import numpy as np
from kmeans_pytorch import kmeans
from xgboost import XGBClassifier

import al_ma_thesis_tjong.presets.dataset_preset as datasets_preset
from al_ma_thesis_tjong.presets import qbc_preset as qbc_preset

'''
Train ensemble xgboost using QBC, the main goal is to use minimum amount of data at the last batch iteration
Also assisted using diversity analysis on data

batch query strategy:
    1. 60% highest entropy from every cluster
    2. 20% the rest with highest entropy that hasn't been chosen
    3. 20% random, necessary to ensure every model is different

    idx_ratio = [1. , 2. , 3. ]
First training iteration: random data with batch size
n-th training iteration: using idx_ratio
Training progress done until certain amount of data is used (n_data)
'''


def qbc(dataset):
    # parameters
    n_model = 20
    dataset = dataset.lower()

    # paths
    model_path = os.path.join(dr(dr(abspath(__file__))), 'results', dataset)
    csv_path = os.path.join(model_path, 'xgb_qbc.csv')

    # CUDA
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    device_cpu = torch.device("cpu")
    dataloader_kwargs = {'pin_memory': True} if cuda_flag else {}
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # load dataset
    if dataset == 'reduced_f_mnist':
        data_train, target_train = datasets_preset.provide_reduced_f_mnist(train=True)
        data_test, target_test = datasets_preset.provide_reduced_f_mnist(train=False)
    elif dataset == 'reduced_mnist':
        data_train, target_train = datasets_preset.provide_reduced_mnist(train=True)
        data_test, target_test = datasets_preset.provide_reduced_mnist(train=False)
    elif dataset == 'unreduced_f_mnist':
        data_train, target_train = datasets_preset.provide_unreduced_f_mnist(train=True)
        data_test, target_test = datasets_preset.provide_unreduced_f_mnist(train=False)
    elif dataset == 'unreduced_mnist':
        data_train, target_train = datasets_preset.provide_unreduced_mnist(train=True)
        data_test, target_test = datasets_preset.provide_unreduced_mnist(train=False)

    # load index
    train_index = pickle.load(open( 'C:\\Users\\steve\\Desktop\\projects\\active-learning-prototypes\\results\\'
                           'unreduced_mnist\\run_001\\indices_batch_019.pkl', 'rb'))
    get_from = -30001
    train_index = train_index[1][get_from:-1]
    random_index = np.random.randint(0, 60000, len(train_index))

    models = []
    tree_method = "auto"  # "gpu_hist" if cuda_flag else "auto"
    print('Tree creation method: ' + tree_method)
    xgbc = XGBClassifier(max_depth=8, objective='objective=multi:softmax', n_estimators=1, n_jobs=32,
                         reg_lambda=1, gamma=2, learning_rate=1, num_classes=10, tree_method=tree_method)
    xgbc.fit(data_train[train_index], target_train[train_index])
    models.append(xgbc)

    xgbc = XGBClassifier(max_depth=8, objective='objective=multi:softmax', n_estimators=1, n_jobs=32,
                         reg_lambda=1, gamma=2, learning_rate=1, num_classes=10, tree_method=tree_method)
    xgbc.fit(data_train[random_index], target_train[random_index])
    models.append(xgbc)

    # training and test process, 1st batch
    output_list_test = np.zeros((2, data_test.__len__())).astype(int)  # n_models x n_data x n_classes
    for i_model in range(2):
        output_list_test[i_model, :] = models[i_model].predict(data_test)

    # Document first batch
    acc_models = qbc_preset.each_model_acc(output_list_test, target_test)
    print(acc_models)
    a = 1


if __name__ == '__main__':
    qbc('unreduced_mnist')