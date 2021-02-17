import os
from os.path import dirname as dr, abspath
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from al_ma_thesis_tjong.presets.dataset_preset import Dataset_MNIST_n
from al_ma_thesis_tjong.presets.models_preset import Encoder, Autoencoder

'''
This file create a trained MNIST, which can be used to reduce MNIST data
'''


def main():
    # training param
    batch_train = 64
    batch_test = 2000
    lr = 1e-3
    epochs = 4
    momentum = 0.9
    torch.manual_seed(1)  # reset random for reproducibility
    save_model = True
    model_path = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'results', 'mnist', 'Autoencoder_mnist.pt')
    output_path = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'results', 'mnist', 'AE_output_mnist.pt')
    input_path = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'results', 'mnist', 'AE_input_mnist.pt')
    mnist_root = os.path.join(dr(dr(dr(dr(abspath(__file__))))), 'data')

    # CUDA
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    device_cpu = torch.device("cpu")
    dataloader_kwargs = {'pin_memory': True} if cuda_flag else {}
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # about datasets
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = Dataset_MNIST_n(root=mnist_root, train=True, download=True, transform=transform, n=60000)
    test_dataset = Dataset_MNIST_n(root=mnist_root, train=False, download=True, transform=transform, n=10000)

    train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=True, **dataloader_kwargs)

    model = Autoencoder().to(device) # define model

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # train autoencoder
    for i_epoch in range(epochs):
        print("Epoch: " + str(i_epoch))
        model.train()
        for _, (data, _, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            print('Train autoencoder, loss = ' + str(loss.item()))

    # store some output of trained AE, to view the result
    data_test, _, _ = next(iter(test_loader))
    data_test = data_test.view(-1, 784).to(device)
    model.eval()
    output = model(data_test)

    # all to cpu
    data_test = data_test.to(device_cpu)
    model = model.to(device_cpu)
    output = output.to(device_cpu)

    # save output
    if save_model:
        torch.save(model.state_dict(), model_path)
        torch.save(output, output_path)
        torch.save(data_test, input_path)


if __name__ == '__main__':
    main()