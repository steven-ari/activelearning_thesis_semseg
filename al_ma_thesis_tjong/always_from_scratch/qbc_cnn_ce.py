import os
from os.path import dirname as dr, abspath
import csv
import pickle
import random
import statistics
import math
import numpy as np
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, BatchSampler, SubsetRandomSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import al_ma_thesis_tjong.presets.dataset_preset as datasets_preset
from al_ma_thesis_tjong.presets import qbc_preset as qbc_preset


# https://github.com/pytorch/examples/blob/master/mnist/main.py
class Cnn_model(nn.Module):
    def __init__(self):
        super(Cnn_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def restore_dropout(fcn_model, orig_prob_list):
    for each_module in fcn_model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.eval()
            each_module.p = orig_prob_list[0]
            orig_prob_list.pop(0)
    return fcn_model


# apply dropout on line 144/145/146, but later is better
# To apply dropout even during .eval()
def enable_dropout(fcn_model, p):
    prob_list = []
    for each_module in fcn_model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
            prob_list.append(each_module.p)
            each_module.p = p
    return fcn_model, prob_list


def train_batch(fcn_model, data_train, target_train, optimizer, device, criterion):
    # train
    '''if torch.cuda.device_count() > 1:
        fcn_model = nn.DataParallel(fcn_model)'''
    fcn_model.train()
    data, target = data_train.cuda(), target_train.cuda()
    optimizer.zero_grad()
    output = fcn_model(data)

    # backpropagate
    # output = {'out': output}
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # tidy up
    loss_np = loss.item()
    fcn_model.eval()
    # fcn_model_cpu = fcn_model.to('cpu')
    del data, target, output, loss
    if device.type == "cuda":
        print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3))
              + ' Gb')  # show gpu usage
        print("Max Memory allocated:" + str(np.round(torch.cuda.max_memory_allocated(device) / 1e9, 3))
              + ' Gb')  # show gpu usage

    return loss_np, fcn_model, optimizer


def model_test(model, data_test, target_test):
    # torch.cuda.empty_cache()
    with torch.no_grad():
        data = data_test.cuda()
        model.cuda()
        model.eval()
        data.cuda()
        output = model(data).detach().cpu().numpy().argmax(axis=1)

        acc = np.sum(output == np.array(target_test))/target_test.__len__()

    return acc


def entropy_dropout_mnist(fcn_model, train_dataset, train_index, device, n_model, dropout_rate, batch_test_size,
                          i_epoch, n_data):
    sample_train, _, _ = train_dataset.__getitem__(1)
    ce_train_array = np.zeros(shape=train_dataset.__len__())
    fcn_model.eval()

    fcn_model, orig_prob_list = enable_dropout(fcn_model, dropout_rate)
    print('After enable: ')

    # ve will be calculated only on data of this indices
    calc_ve_idx = np.array(range(train_dataset.__len__()))
    calc_ve_idx = calc_ve_idx[np.isin(calc_ve_idx, train_index, invert=True)]
    print('CE of ' + str(np.unique(calc_ve_idx).__len__()) + ' data')
    print('Trained length: ' + str(np.unique(train_index).__len__()) + ' data')

    with torch.no_grad():
        batch_sampler = BatchSampler(SubsetRandomSampler(calc_ve_idx), batch_size=batch_test_size, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)  # calculate vote entropy in 25% increment
        for i_batch, (data, target, index) in enumerate(train_dataloader):
            if device.type == "cuda":  # show gpu usage
                print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3)) + ' Gb')
            data_length = data.shape[0]
            output_list_train = np.zeros((n_model, data_length, 10))
            for i_model in range(n_model):
                print('Model:' + str(i_model) + ', batch: ' + str(i_batch+1) + '/' + str(train_dataloader.__len__()) +
                      ', epoch: ' + str(i_epoch) + ', total VE: ' + str(len(calc_ve_idx)))
                print('Shape data:' + str(data.shape))
                output_list_train[i_model] = fcn_model(data).detach().cpu().numpy()
                print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3)) + ' Gb')
                print(output_list_train[i_model].mean())
            del data, target

            # calculate consensus entropy
            output_list_train = np.exp(output_list_train)
            # log_softmax to probability
            output_list_train = (output_list_train / output_list_train.sum(axis=2, keepdims=True))
            ce_batch = entropy(np.sum(output_list_train, axis=0), base=2, axis=1)
            ce_train_array[index] = ce_batch

        if device.type == "cuda":  # show gpu usage
            print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3)) + ' Gb')

    fcn_model = restore_dropout(fcn_model, orig_prob_list)
    # ve_train_array: array for easier processing, only one value for each data, which is average ve on every pixels
    ve_sorted = np.argsort(ce_train_array)[::-1]
    indices = np.setdiff1d(ve_sorted, train_index, assume_unique=True)[0:n_data]

    return indices, ce_train_array


# qcb vote vs qbc consensus vs random, every qbc with dropout
def qbc(n_model, n_train, qbc_batch_size, batch_size, idx_ratio, dataset, seed):
    # parameters
    n_model = n_model
    n_train = n_train
    dataset = dataset.lower()  # 'reduced_f_mnist', 'reduced_mnist','unreduced_f_mnist','unreduced_mnist',
    qbc_batch_size = qbc_batch_size
    batch_train_size = batch_size
    batch_test_size = 10000
    lr = 0.0001
    test_factor = 5  # committee only tested every test_factor-th batch
    poly_exp = 1.0
    n_epoch = 20
    dropout_rate = 0.25
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # paths
    result_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', dataset, 'always_from_scratch', 'cnn_qbc')
    csv_path = os.path.join(result_path, 'cnn_qbc.csv')
    csv_name_train = 'train.csv'
    csv_name_test = 'test.csv'
    csv_name_index = 'index.csv'
    csv_name_index_compare = 'index_ce.csv'
    mnist_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data')
    index_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', dataset, 'always_from_scratch', 'cnn_qbc')

    # CUDA
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    device_cpu = torch.device("cpu")
    dataloader_kwargs = {'pin_memory': True} if cuda_flag else {}
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    torch.cuda.empty_cache()
    kwargs = {'num_workers': 4, 'pin_memory': True, 'shuffle': True}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets_preset.Dataset_MNIST_n(root=mnist_path, train=True, transform=transform, n=n_train)
    test_dataset = datasets_preset.Dataset_MNIST_n(root=mnist_path, train=False, transform=transform, n=10000)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_test_size, **kwargs)
    data_test, target_test, index = next(iter(test_dataloader))

    # to document training process, create directory, etc
    text = (('CE: n_model: ' + str(n_model)) + (', n_train: ' + str(n_train)) + (', batch_size: ' + str(batch_size))
            + (', idx_ratio: ' + str(idx_ratio)) + (', dataset: ' + dataset))
    print(text)
    dir_name = 'cnn_'
    dir_number = 1
    while os.path.exists(os.path.join(result_path, (dir_name + '{:03d}'.format(dir_number)))):
        dir_number += 1
    run_path = os.path.join(result_path, (dir_name + '{:03d}'.format(dir_number)))
    os.makedirs(run_path)  # make run_* dir
    f = open(os.path.join(run_path, 'info.txt'), 'w+')  # write .txt file
    f.write(text)
    f.close()

    # model
    model = Cnn_model()
    print("CNN model created")

    model = model.cuda()
    model = nn.DataParallel(model)

    # the optimizers
    optimizer = torch.optim.Adam(model.module.parameters(), lr=lr)
    lambda1 = lambda epoch: math.pow(1 - epoch / n_epoch, poly_exp)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion = F.nll_loss

    # report everything
    print(str(n_model) + " fcn models created")
    text = ('CE: n_model: ' + str(n_model)) + (', n_train: ' + str(n_train)) + (', n_epoch: ' + str(n_epoch)) + \
           (', batch_train_size: ' + str(batch_train_size)) + (', idx_ratio: ' + str(idx_ratio))
    print(text)

    # to document training process, create directory, etc
    train_text = [str(x) for x in range(1, n_epoch + 1)]
    test_text = [str(x) for x in range(1, n_epoch + 1)]
    train_index = np.array([]).astype(np.int16)
    run_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', dataset, 'always_from_scratch', 'cnn_qbc')

    # write training progress
    csv_path_train = os.path.join(run_path, csv_name_train)
    title = ["CE: Training progress for n_model = " + str(n_model) + ", idx_ratio:  " + str(idx_ratio) + ', for multiple epoch']
    with open(csv_path_train, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # write test progress
    csv_path_test = os.path.join(run_path, csv_name_test)
    title = ["CE: Test progress for n_model = " + str(n_model) + ", idx_ratio:  " + str(idx_ratio) + ', for multiple epoch']
    with open(csv_path_test, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # training start
    for i_epoch in range(n_epoch):
        # initialize with random
        if len(train_index) == 0:
            '''train_index = np.array(random.sample(range(n_train), qbc_batch_size))
            trained_index = train_index'''
            with open(os.path.join(index_path, csv_name_index), 'r') as csv_file:
                data = csv_file.readlines()
            train_index = np.array(list(map(int, data[-1][3:-1].split(','))))  # take last index list (last line in csv)
            trained_index = train_index
            index_text = ['One_experiment']
            csv_path_index = os.path.join(index_path, csv_name_index_compare)
            with open(csv_path_index, mode='a+', newline='') as test_file:
                test_writer = csv.writer(test_file, delimiter=';')
                test_writer.writerow(index_text)

        # append with vote entropy
        elif len(train_index) < int(n_train):
            # perform vote entropy on entire dataset
            train_index, ce = entropy_dropout_mnist(model, train_dataset, trained_index, device, n_model,
                                                dropout_rate, batch_test_size, i_epoch, n_data=qbc_batch_size)
            trained_index = np.append(trained_index, train_index)

            index_text = str(ce[train_index[0]]) + ': ' + str([x for x in train_index]).strip('[]')
            index_text = index_text.split(';')
            csv_path_index = os.path.join(index_path, csv_name_index_compare)
            with open(csv_path_index, mode='a+', newline='') as test_file:
                test_writer = csv.writer(test_file, delimiter=';')
                test_writer.writerow(index_text)
        # retrain with selected data
        print(train_index)
        print('length: ' + str(len(train_index)))
        train_subset = Subset(train_dataset, trained_index)
        train_dataloader = DataLoader(train_subset, batch_size=batch_train_size, shuffle=True)
        loss_epoch = []
        model = Cnn_model()
        model = model.cuda()
        model = nn.DataParallel(model)

        # the optimizers
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr)
        lambda1 = lambda epoch: math.pow(1 - epoch / n_epoch, poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        criterion = F.nll_loss
        for i_batch, (data_train, target_train, index) in enumerate(train_dataloader):
            # train batch
            loss, model, optimizer = train_batch(model, data_train, target_train, optimizer, device, criterion)
            print('Epoch: ' + str(i_epoch) + '\t Batch: ' + str(i_batch) + '/' + str(len(train_dataloader))
                  + '; model ' + str(0) + '; train loss avg: ' + "{:.3f}".format(loss))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            loss_epoch.append(loss)

        # document train result
        train_text[i_epoch] = train_text[i_epoch] + ";{:.4f}".format(np.array(loss_epoch).mean()) + \
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr'])) \
                              + ';' + str(len(trained_index))
        # update train documentation
        text = train_text[i_epoch].split(";")
        with open(csv_path_train, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        # save temporary model and perform test
        print('Save and Test Model')
        '''model.train()
        torch.save(model.state_dict(), os.path.join(run_path, ('model_weight_epoch_train' +
                                                                   '{:03d}'.format(i_epoch) + '.pt')))
        model.eval()
        torch.save(model.state_dict(), os.path.join(run_path, ('model_weight_epoch_' +
                                                                   '{:03d}'.format(i_epoch) + '.pt')))'''

        # test
        acc = model_test(model, data_test, target_test)

        # document test result
        test_text[i_epoch] = test_text[i_epoch] + ";{:.4f}".format(acc) + \
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr'])) \
                              + ';' + str(len(trained_index))
        print("Acc: " + "{:.4f}".format(acc))

        # update test documentation
        text = test_text[i_epoch].split(";")
        with open(csv_path_test, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        # one epoch ends here
        # scheduler.step()
        print(optimizer)


if __name__ == '__main__':
    qbc(n_model=5, n_train=60000, qbc_batch_size=3000, batch_size=60, idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist')