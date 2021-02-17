import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import statistics
import csv
from os.path import dirname as dr, abspath

import torch
from concurrent import futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

from al_ma_thesis_tjong.presets import segmen_preset as segmen_preset


# First average the probability of each class from model votes
def committee_vote(output_list, target):
    target_stack = np.stack([target]*output_list.shape[0], axis=0)
    vote_each_data = np.sum(output_list == target_stack, axis=0)  # how many models got it correct
    acc_all = (vote_each_data/output_list.shape[0]).mean()
    return acc_all


# test accuracy of each model in committee, accuracy shape length of committee
def each_model_acc(output_list, target):
    target_stack = np.stack([target]*output_list.shape[0], axis=0)
    votes = np.sum(output_list == target_stack, axis=1)
    acc_models = votes / output_list.shape[1]
    return acc_models


# get voting accuracy out of multiple models, accuracy shape length of target
def each_target_acc(output_list, target):
    target_stack = np.stack([target] * output_list.shape[0], axis=0)
    acc_target = np.sum(target_stack == output_list, axis=0)/output_list.shape[0]
    return acc_target


# Vote entropy as defined in Active Learning Literature Survey [Settles,2009]
def vote_entropy(output_list):
    v_yi_c = np.exp(np.sum(output_list, axis=1)/output_list.shape[1])
    result = entropy(v_yi_c, base=2, axis=1)  # *10 to scale with sum entropy
    return result


# Vote entropy as defined in Active Learning Literature Survey [Settles,2009]
def vote_entropy_xgb(output_list, target):
    labels = np.unique(target)
    v_yi_c = np.zeros((output_list.shape[1], labels.__len__()))
    for i_label in range(labels.__len__()):
        label_stack = np.ones_like(output_list)*labels[i_label]
        v_yi_c[:, i_label] = np.sum(output_list == label_stack, axis=0)/output_list.shape[0]

    result = entropy(v_yi_c, base=2, axis=1)
    return result


# get indices of next training batch
def get_next_indices(idx_library, entropy, idx_data_cr, batch_size, idx_ratio, data_all_len):
    '''
    random data will be added directly at training
    :param data_all_len: length or training dataset
    :param idx_library: index of data already used for training
    :param entropy: entropy values
    :param idx_data_cr: index of data in every cluster
    :param batch_size: number of data added to next batch
    :param idx_ratio: ratio to decide how data for next batch added
    :return:
    '''
    indices = np.array([]).astype(int)
    indices_en = np.argsort((-1) * entropy)
    indices_en = indices_en[np.isin(indices_en, np.array(idx_library), invert=True)] # exclude used data
    n_each_cluster = math.floor(idx_ratio[0]*batch_size/idx_data_cr.__len__())
    # append from each cluster
    for i_cluster in range(idx_data_cr.__len__()):
        from_cr = indices_en[np.in1d(indices_en, idx_data_cr[i_cluster])][0:n_each_cluster]
        indices = np.append(indices, from_cr)
    print("Diversity: " + str(indices.__len__()))
    # append highest entropy not yet chosen
    n_highest_en = (int((idx_ratio[0]+idx_ratio[1])*batch_size)-indices.__len__())
    from_highest_en = indices_en[np.isin(indices_en, np.array(indices), invert=True)][0:n_highest_en]
    indices = np.append(indices, from_highest_en)
    print("Entropy: " + str(from_highest_en.__len__()))

    # add random index
    random_suggest = np.array(random.sample(range(data_all_len), (batch_size + idx_library.__len__())))   # create random
    random_suggest = random_suggest[np.isin(random_suggest, np.append(idx_library, indices), invert=True)]  # exclude in quarantine
    random_suggest = random_suggest[0:int(batch_size * idx_ratio[2])]
    indices = np.append(indices, random_suggest)  # append to training index
    print("Random: " + str(random_suggest.__len__()))

    idx_library = np.append(idx_library, indices)

    return idx_library


def get_entropy_acc(entropy, output_list, target):
    csv_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've.csv')
    rand_idx = np.random.randint(low=0, high=60000, size=1000)
    train_text = 0
    rev_entropy = np.around(entropy[rand_idx], decimals=3)

    # calculate accuracy
    rev_output = output_list[:, rand_idx]
    rev_target = target[rand_idx]
    target_stack = np.array([rev_target] * output_list.shape[0])
    rev_acc = np.sum(rev_output == target_stack, axis=0)/output_list.shape[0]

    # np.savetxt(csv_path, rev_acc, delimiter=";")
    with open(csv_path, mode='a+') as test_file:
        test_writer = csv.writer(test_file, delimiter=';')
        test_writer.writerow(rev_acc)
        test_writer.writerow(rev_entropy)


# plot n-highest or n-lowest entropy
def show_entropy_result(acc_models, entropy, output_list, data_train, target_train):
    plot_path = '/Users/steven_ari/Desktop/active-learning-prototypes/results/plots/entropy/good_result'
    random_idx = np.argsort((-1) * entropy)[-100:]  # only with n-lowest entropy

    plot_path = '/Users/steven_ari/Desktop/active-learning-prototypes/results/plots/entropy/confusing_result'
    random_idx = np.argsort((-1) * entropy)[0:30]  # only with n-highest entropy

    my_dpi = 100

    # show single pictures
    for i_single in range(random_idx.__len__()):
        idx = random_idx[i_single]
        # show without text
        plt.figure(figsize=(700 / my_dpi, 700 / my_dpi), dpi=my_dpi)
        plt.imshow(data_train.reshape(data_train.__len__(), 28, 28)[idx], cmap='gray')
        text = ("Entropy: " + "{:.3f}".format(entropy[idx]))
        plt.title(text, fontsize=16)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(fname=os.path.join(plot_path, ('plain_en_' + str(i_single) + '.png')), format='png', dpi=my_dpi * 3)

        # show with text
        # show entropy, show committee acc, 3 highest guess, entropy value, show 8 of it?
        plt.figure(figsize=(700/my_dpi, 700/my_dpi), dpi=my_dpi)
        plt.imshow(data_train.reshape(data_train.__len__(), 28, 28)[idx], cmap='gray')
        predict_all = (np.sum(output_list[:, idx, :], axis=0)/20)*100
        high_guess_idx = np.argsort((-1) * predict_all)[0:3]
        high_guess = predict_all[high_guess_idx]
        text = ("Entropy: " + "{:.3f}".format(entropy[idx]) +
                "; acc:" + "{:.2f}".format(predict_all[target_train[idx]]) +
                "%; ground truth: \"" + str(target_train[idx]) +
                "\"\n 2 highest guess: \"" +
                str(high_guess_idx[0]) + "\"=" + "{:.2f}".format(high_guess[0]) + "%, \"" +
                str(high_guess_idx[1]) + "\"=" + "{:.2f}".format(high_guess[1]) + "%")
        plt.title(text,fontsize=16)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(fname=os.path.join(plot_path, ('text_en_' + str(i_single) + '.png')), format='png', dpi=my_dpi * 3)


def plot_ugly(output_list_train, data_train, target_train):
    # index for high entropy for committee high acc
    idx = [500, 10116, 15434, 20773, 25562, 25678, 26560, 28620, 36104, 39184, 41594, 42566, 45352, 47034, 50329, 51248,
           51794, 52086]

    for i_data in range(idx.__len__()):

        # plot titleless
        plt.imshow(data_train.reshape(data_train.__len__(), 28, 28)[idx[i_data]], cmap='gray')
        pred = np.bincount(output_list_train[:, idx[i_data]]).argmax()
        print(target_train[idx[i_data]])
        print(pred)
        name = 'titleless_' + str(i_data)
        plt.title('')
        plot_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'malicious')
        file_name = os.path.join(plot_path, (name + '.png'))
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.tight_layout()
        plt.savefig(file_name, format='png', dpi=300)

        # plot with title
        plt.clf()
        plt.imshow(data_train.reshape(data_train.__len__(), 28, 28)[idx[i_data]], cmap='gray')
        pred = np.bincount(output_list_train[:, idx[i_data]]).argmax()
        print(target_train[idx[i_data]])
        print(pred)
        name = 'title_' + str(i_data)
        plt.title('Data: ' + str(idx[i_data]) + '; Prediction: ' + str(pred) + ', GT: '
                  + str(target_train[idx[i_data]]), fontsize=16)
        plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.tight_layout()
        file_name = os.path.join(plot_path, (name + '.png'))
        plt.savefig(file_name, format='png', dpi=300)




def parallel_label_comp(label_num, output_list):

    label_stack = np.ones_like(output_list) * label_num
    v_yi_c_label = np.sum(output_list == label_stack, axis=0)/output_list.shape[0]

    return v_yi_c_label


def vote_entropy_semseg(output_list, labels_valid):
    # output_list: n_models x n_data x H x W
    # v_yi_c.shape: n_data x n_classes x H x W
    # output_list = output_list.argmax(axis=2).astype(np.int8)  # n_models x n_data x H x W

    # prepare parallel vote entropy calculation
    num_cores = multiprocessing.cpu_count()
    v_yi_c_list = []
    with ProcessPoolExecutor(max_workers=1) as executor:
        v_yi_c_all = [executor.submit(parallel_label_comp, x, output_list) for x in list(range(labels_valid.__len__()))]
        for v_yi_c_label in futures.as_completed(v_yi_c_all):
            v_yi_c_list.append(v_yi_c_label.result())
    v_yi_c = np.array(v_yi_c_list)
    # should be: n_data x H x W, then sum for every pixel
    en = entropy(np.array(v_yi_c), base=2, axis=0)  # vote entropy for each pixel in each data
    # average ve on every pixel, shape: length of batch
    result = np.sum(en, axis=(-2, -1))/(output_list.shape[-2]*output_list.shape[-1])
    return result


# using proper numpy implementation, this one is faster
def vote_entropy_semseg_old(output_list, labels_valid):
    # output_list: n_models x n_data x H x W
    # label_stack_list: n_classes x n_models x n_data x H x W

    label_stack_list = []
    for i_label in range(labels_valid.__len__()):
        label_stack = np.ones_like(output_list) * i_label
        label_stack_list.append(label_stack)

    # v_yi_c: n_classes x n_data x H x W
    # expanded_output: n_classes x n_models x n_data x H x W, should have sma shape with label_stack_list
    expanded_output = np.concatenate([np.expand_dims(output_list, 0)] * labels_valid.__len__()).astype(np.int8)
    v_yi_c = (np.sum(expanded_output == np.array(label_stack_list), axis=1).astype(np.float32) / output_list.shape[0])

    # en: n_data x H x W, then sum for every pixel
    en = entropy(v_yi_c, base=2, axis=0)  # vote entropy for each pixel in each data
    # average ve on every pixel, shape: length of batch
    result = np.sum(en, axis=(-2, -1))/(output_list.shape[-2]*output_list.shape[-1])
    return result


def get_next_indices(idx_library, ve_train_all, idx_ratio, batch_train_size, data_train_len):

    indices = np.array([]).astype(int)
    indices_en = np.argsort((-1) * ve_train_all)
    indices_en = indices_en[np.isin(indices_en, np.array(idx_library), invert=True)]  # exclude used data

    # append from highest entropy
    n_highest_en = int(idx_ratio[0] * batch_train_size)
    from_highest_en = indices_en[0:n_highest_en]  # indices_en[0:0] returns empty
    indices = np.append(indices, from_highest_en)
    # print("Entropy: " + str(from_highest_en.__len__()))

    # add random index
    random_suggest = np.array(
        random.sample(range(data_train_len), k=(batch_train_size + idx_library.__len__())))   # create random index
    random_suggest = random_suggest[
        np.isin(random_suggest, np.append(idx_library, indices), invert=True)]  # exclude index already included
    random_suggest = random_suggest[0:int(batch_train_size * idx_ratio[1])]
    indices = np.append(indices, random_suggest)  # append to training index
    # print("Random: " + str(random_suggest.__len__()))

    idx_library = np.append(idx_library, indices)
    # print(idx_library)

    return idx_library


