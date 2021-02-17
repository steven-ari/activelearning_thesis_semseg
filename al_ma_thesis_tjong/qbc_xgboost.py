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


def qbc(n_model, n_train, batch_size, idx_ratio, dataset):
    # parameters
    n_model = n_model
    n_train = n_train
    batch_size = batch_size
    idx_ratio = idx_ratio
    n_cluster = 20
    dataset = dataset.lower()  # 'reduced_f_mnist', 'reduced_mnist','unreduced_f_mnist','unreduced_mnist',
    text = (('n_model: ' + str(n_model)) + (', n_train: ' + str(n_train)) + (', batch_size: ' + str(batch_size))
            + (', idx_ratio: ' + str(idx_ratio)) + (', n_cluster: ' + str(n_cluster)) + (', dataset: ' + dataset))
    print(text)

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

    # execute kmeans-clustering for entire training dataset
    cluster_index, cluster_centers = kmeans(X=torch.from_numpy(data_train),
                                            num_clusters=n_cluster, distance='cosine', device=device)
    # show clustering result, document data per cluster
    n_data_cr = np.zeros(n_cluster, dtype=int)
    idx_data_cr = []
    for i_cluster in range(n_cluster):
        n_data_cr[i_cluster] = np.sum(cluster_index.numpy() == i_cluster)
        idx_data_cr.append(np.argwhere(cluster_index == i_cluster).numpy())
        print("Cluster " + str(i_cluster) + ": " + str(n_data_cr[i_cluster])
              + " data, or " + "{:.4f}".format(n_data_cr[i_cluster] / cluster_index.__len__() * 100) + "%")
    print("Cluster data size variance: " + "{:.4f}".format(n_data_cr.var() ** 0.5) + ", (smaller is better)")

    # to document training process, create directory, etc
    train_text = [str(x) for x in range(batch_size, n_train + 1, batch_size)]
    dir_name = 'run_'
    dir_number = 1
    while os.path.exists(os.path.join(model_path, (dir_name + '{:03d}'.format(dir_number)))):
        dir_number += 1
    run_path = os.path.join(model_path, (dir_name + '{:03d}'.format(dir_number)))
    os.makedirs(run_path)  # make run_* dir
    f = open(os.path.join(run_path, 'info.txt'), 'w+')  # write .txt file
    f.write(text)
    f.close()

    # create models and index library
    models = []
    tree_method = "auto"  # "gpu_hist" if cuda_flag else "auto"
    print('Tree creation method: ' + tree_method)
    idx_library = [np.array([]).astype(int) for x in range(n_model)]
    for i_model in range(n_model):
        xgbc = XGBClassifier(max_depth=8, objective='objective=multi:softmax', n_estimators=1, n_jobs=32,
                             reg_lambda=1, gamma=2, learning_rate=1, num_classes=10, tree_method=tree_method)
        models.append(xgbc)
    print(str(n_model) + " xgboost models created")

    # training and test process, 1st batch
    output_list_test = np.zeros((n_model, data_test.__len__())).astype(int) # n_models x n_data x n_classes
    for i_model in range(n_model):
        random_index = np.array(random.sample(range(data_train.__len__()), k=batch_size))
        idx_library[i_model] = np.append(idx_library[i_model], random_index)
        models[i_model].fit(data_train[random_index], target_train[random_index])
        output_list_test[i_model, :] = models[i_model].predict(data_test)

    # Document first batch
    acc_models = qbc_preset.each_model_acc(output_list_test, target_test)
    acc_committee = qbc_preset.committee_vote(output_list_test, target_test)  # committee vote
    train_text[0] = train_text[0] + ' '.join([";" + "{:.4f}".format(elem) for elem in acc_models])
    train_text[0] = train_text[0] + '; ' + "{:.3f}".format(acc_committee * 100) + '%'  # committee vote
    print("First batch added!")
    print("Batch " + str(0) + ": average acc of models is " + "{:.3f}".format(acc_models.mean() * 100) + "%")
    print("Batch " + str(0) + ": acc of committee is " + "{:.3f}".format(acc_committee * 100) + "%")
    print("Library sizes, after first batch:" + str([np.unique(idx_library[i_model]).shape for x in range(n_model)]))
    pickle.dump(models, open(os.path.join(run_path, ('models_batch_' + "{0:0=3d}".format(0) + '.pkl')), 'wb'))
    pickle.dump(idx_library, open(os.path.join(run_path, ('indices_batch_' + "{0:0=3d}".format(0) + '.pkl')), 'wb'))

    # training process, n-th batch
    for i_batch in range(1, train_text.__len__()):
        print("Starting Batch " + str(i_batch))
        output_list_train = np.zeros((n_model, data_train.__len__())).astype(int)

        # calculate entropy & acc of current data
        for i_model in range(n_model):
            output_list_train[i_model, :] = models[i_model].predict(data_train)
        acc_models = qbc_preset.each_model_acc(output_list_train, target_train)
        acc_target = qbc_preset.each_target_acc(output_list_train, target_train)
        entropy = qbc_preset.vote_entropy_xgb(output_list_train, target_train)
        # qbc_preset.get_entropy_acc(entropy, output_list_train, target_train)
        # show entropy, show committee acc, 3 highest guess, entropy value, show 8 of it?
        # qbc_preset.show_entropy_result(acc_models, entropy, output_list, data_train, target_train)
        # qbc_preset.plot_ugly(output_list_train, data_train, target_train)
        print("Library sizes:" + str([np.unique(idx_library[i_model]).shape for x in range(n_model)])) 
        index_1 = np.random.choice(range(n_model))
        index_2 = np.random.choice(np.setdiff1d(range(0, n_model), index_1))
        print("Overlap size:" + str(np.intersect1d(idx_library[index_1], idx_library[index_2]).__len__()) +
              ", overlap ideal: " + str(int((idx_library[index_2].__len__() - batch_size)
                                            * (idx_ratio[0] + idx_ratio[1]))) +
              ", library size: " + str(idx_library[index_2].__len__()) + ", dataset: " + dataset
              + ", idx_ratio: " + str(idx_ratio))

        # train and test for each model and each batch
        for i_model in range(n_model):
            # indexes
            idx_library[i_model] = \
                qbc_preset.get_next_indices(idx_library[i_model], entropy, idx_data_cr, batch_size,
                                            idx_ratio, data_train.__len__())
            # train model
            models[i_model].fit(data_train[idx_library[i_model]], target_train[idx_library[i_model]])
            # test model
            output_list_test[i_model, :] = models[i_model].predict(data_test)
            print('Model ' + str(i_model))

        # check committee vote
        acc_models = qbc_preset.each_model_acc(output_list_test, target_test)
        acc_committee = qbc_preset.committee_vote(output_list_test, target_test)  # committee vote method
        print("Batch " + str(i_batch) + ": average acc of models is " + "{:.3f}".format(acc_models.mean() * 100) + "%")
        print("Batch " + str(i_batch) + ": acc of committee is " + "{:.3f}".format(acc_committee * 100) + "%")

        # Document training progress
        train_text[i_batch] = train_text[i_batch] + ' '.join([";" + "{:.4f}".format(elem) for elem in acc_models])
        train_text[i_batch] = train_text[i_batch] + '; ' + "{:.3f}".format(
            acc_committee * 100) + '%'  # committee vote method
        # save models and indices
        pickle.dump(models, open(os.path.join(run_path, ('models_batch_' + "{0:0=3d}".format(i_batch) + '.pkl')), 'wb'))
        pickle.dump(idx_library,
                    open(os.path.join(run_path, ('indices_batch_' + "{0:0=3d}".format(i_batch) + '.pkl')), 'wb'))

    # write text to csv
    title = ["New Vote, Results for n_model = " + str(n_model) + ", idx_ratio:  " + str(idx_ratio)
             + ", n_cluster: " + str(n_cluster) + ", with highest entropy, avg and var documented"]
    with open(csv_path, mode='a+') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)
    # loop through train_text
    for i_text in range(0, train_text.__len__()):
        text = train_text[i_text].split(";")
        mean = statistics.mean([float(i) for i in text[1:-2]])
        var = statistics.variance([float(i) for i in text[1:-2]]) ** 0.5
        text.append("{:.3f}".format(mean * 100) + "%")
        text.append("{:.3f}".format(var * 100) + "%")
        with open(csv_path, mode='a+') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)
