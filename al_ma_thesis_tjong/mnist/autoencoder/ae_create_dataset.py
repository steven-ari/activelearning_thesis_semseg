import os
from os.path import dirname as dr, abspath
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from al_ma_thesis_tjong.presets.dataset_preset import Dataset_MNIST_n
from al_ma_thesis_tjong.presets.models_preset import Encoder, Autoencoder


'''
Convert standard MNIST Dataset into reduced MNIST using Autoencoder
'''

def main():
    # training param
    batch_train = 64
    batch_test = 2000
    lr = 1e-3
    epochs = 4
    momentum = 0.9
    torch.manual_seed(1)  # reset random for reproducibility
    model_path = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'results', 'mnist', 'Autoencoder_mnist.pt')
    train_path = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'data', 'Dataset_MNIST_n', 'coded_mnist_train.pt')
    test_path = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'data', 'Dataset_MNIST_n', 'coded_mnist_test.pt')
    mnist_root = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'data')

    # CUDA
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    device_cpu = torch.device("cpu")
    dataloader_kwargs = {'pin_memory': True} if cuda_flag else {}
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = Dataset_MNIST_n(root=mnist_root, train=True, download=True, transform=transform, n=60000)
    test_dataset = Dataset_MNIST_n(root=mnist_root, train=False, download=True, transform=transform, n=10000)
    train_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

    # model
    saved_weights = torch.load(model_path, map_location=torch.device('cpu'))  # load saved weights
    model_weights = {"encoder_in.weight": saved_weights["encoder_in.weight"],  # load weight only from encoder part
                     "encoder_hidden.weight": saved_weights["encoder_hidden.weight"],
                     "encoder_in.bias": saved_weights["encoder_in.bias"],
                     "encoder_hidden.bias": saved_weights["encoder_hidden.bias"]}
    model = Encoder()
    model.load_state_dict(model_weights)
    model = nn.DataParallel(model.to(device))  # DataParallel allow using multiple GPU

    # code training and test data
    with torch.no_grad():  # stop tracking var gradient to reduce comp cost
        # convert train MNIST
        data, target, index = next(iter(train_loader))
        data = data.view(-1, 784).to(device)
        output = model(data)
        output = output.to(device_cpu)
        print("Training data converted")
        print(index)
        torch.save((output, target, index), train_path)  # save converted data

        # convert test MNIST
        data, target, index = next(iter(test_loader))
        data = data.view(-1, 784).to(device)
        output = model(data)
        output = output.to(device_cpu)
        print("Test data converted")
        print(index)
        torch.save((output, target, index), test_path)  # save converted data


if __name__ == '__main__':
    main()