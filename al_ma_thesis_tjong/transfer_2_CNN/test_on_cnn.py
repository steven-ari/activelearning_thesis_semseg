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
        del data
    return acc


# qcb vote vs qbc consensus vs random, every qbc with dropout
def cart_2_cnn(dataset):
    # parameters
    al_batch = 3000
    dataset = dataset.lower()  # 'reduced_f_mnist', 'reduced_mnist','unreduced_f_mnist','unreduced_mnist',
    batch_train_size = 60
    batch_test_size = 10000
    n_train = 60000
    lr = 0.0001
    test_factor = 5  # committee only tested every test_factor-th batch
    poly_exp = 1.0
    n_epoch = 20
    dropout_rate = 0.25

    # seed
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # paths
    result_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', dataset, 'from_cart', 'cnn_qbc')
    csv_path = os.path.join(result_path, 'cnn_qbc.csv')
    csv_name_train = 'train.csv'
    csv_name_test = 'test.csv'
    csv_name_index = 'index.csv'
    mnist_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data')

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
    dir_name = 'cnn_from_cart_'
    dir_number = 1
    while os.path.exists(os.path.join(result_path, (dir_name + '{:03d}'.format(dir_number)))):
        dir_number += 1
    run_path = os.path.join(result_path, (dir_name + '{:03d}'.format(dir_number)))
    os.makedirs(run_path)  # make run_* dir
    f = open(os.path.join(run_path, 'info.txt'), 'w+')  # write .txt file
    f.close()

    # to document training process, create directory, etc
    train_text = [str(x) for x in range(1, n_epoch + 1)]
    test_text = [str(x) for x in range(1, n_epoch + 1)]

    # write training progress
    csv_path_train = os.path.join(run_path, csv_name_train)
    title = ["Random: Training progress for n_model = " + str(1) + ", idx_ratio:  " + str('none') + ', for multiple epoch']
    with open(csv_path_train, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # write test progress
    csv_path_test = os.path.join(run_path, csv_name_test)
    title = ["Random: Test progress for n_model = " + str(1) + ", idx_ratio:  " + str('none') + ', for multiple epoch']
    with open(csv_path_test, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # read indices
    index_library = pickle.load(open("C:\\Users\\steve\\Desktop\\projects_software\\active-learning-prototypes\\results\\unreduced_mnist\\run_013\\indices_batch_019.pkl", "rb"))
    index_library = index_library[0]

    for i_al_batch in range(int(len(index_library)/al_batch)):
        index_batch = index_library[0: (i_al_batch+1)*al_batch]
        train_subset = Subset(train_dataset, index_batch)
        train_dataloader = DataLoader(train_subset, batch_size=batch_train_size, shuffle=True)
        loss_epoch = []

        # model
        model = Cnn_model()
        model = model.cuda()
        model = nn.DataParallel(model)
        print("CNN model created")

        # the optimizers
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr)
        lambda1 = lambda epoch: math.pow(1 - epoch / n_epoch, poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        criterion = F.nll_loss
        for i_batch, (data_train, target_train, index) in enumerate(train_dataloader):
            # train batch
            loss, model, optimizer = train_batch(model, data_train, target_train, optimizer, device, criterion)
            print('Epoch: ' + str(i_batch) + '\t Batch: ' + str(i_batch) + '/' + str(len(train_dataloader))
                  + '; model ' + str(0) + '; train loss avg: ' + "{:.3f}".format(loss))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            loss_epoch.append(loss)

        # document train result
        train_text[i_al_batch] = train_text[i_al_batch] + ";{:.4f}".format(np.array(i_al_batch).mean()) + \
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr'])) \
                              + ';' + str(len(index_batch))
        # update train documentation
        text = train_text[i_al_batch].split(";")
        with open(csv_path_train, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)# test
        acc = model_test(model, data_test, target_test)

        # document test result
        test_text[i_al_batch] = test_text[i_al_batch] + ";{:.4f}".format(acc) + \
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr'])) \
                              + ';' + str(len(index_batch))
        print("Acc: " + "{:.4f}".format(acc))

        # update test documentation
        text = test_text[i_al_batch].split(";")
        with open(csv_path_test, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        # one epoch ends here
        # scheduler.step()
        print(optimizer)
        del model


if __name__ == '__main__':
    cart_2_cnn(dataset='unreduced_mnist')