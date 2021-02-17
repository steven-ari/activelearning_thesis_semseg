import torch
from torch.utils.data import DataLoader
import numpy as np

from al_ma_thesis_tjong.presets.dataset_preset import Reduced_MNIST

# https://github.com/subhadarship/kmeans_pytorch/blob/master/kmeans_pytorch/__init__.py
from kmeans_pytorch import kmeans

'''
apply k-means on reduced MNIST and check if Coded_MNIST works properly
'''


# Provide sse for kmeans, possibly other criteria in the future
def eval_kmeans(data, index, center_all, device):
    data, index, center_all = data.to(device), index.to(device), center_all.to(device)
    sse = np.zeros(center_all.shape[0])

    for i_centers in range(center_all.shape[0]):
        center = center_all[i_centers].repeat(torch.sum(index==i_centers), 1)
        data_index = data[index == i_centers]
        sse[i_centers] = torch.sum((center-data_index)**2).item()/index.__len__()

    return sse.sum()


def main():
    batch_train = 60000
    batch_test = 1000
    attempt_kmeans = 3  # number of k-means attempt
    centers = range(2, 31, 2)  # number of centroids to be tested
    mnist_root = '../../../data'

    # CUDA
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    device_cpu = torch.device("cpu")
    dataloader_kwargs = {'pin_memory': True} if cuda_flag else {}
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # dataset
    train_dataset = Reduced_MNIST(root=mnist_root, train=True)
    test_dataset = Reduced_MNIST(root=mnist_root, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=True)

    data_train, index_train = next(iter(train_loader))
    # plt.imshow((data_train * 0.3081 + 0.1307).view(data_train.__len__(), 28, 28).detach().numpy()[0], cmap='gray')
    print(index_train)
    data_test, index_test = next(iter(test_loader))
    # plt.imshow((data_test * 0.3081 + 0.1307).view(data_test.__len__(), 28, 28).detach().numpy()[0], cmap='gray')
    print(index_test)
    sse_all = []
    sse_temp = np.zeros(attempt_kmeans)

    # execute k-means clustering
    with torch.no_grad():
        data = data_train  # reverse normalization to get original MNIST
        for i_centers in centers:
            for i_loop in range(attempt_kmeans):
                cluster_index, cluster_centers = kmeans(X=data, num_clusters=i_centers, distance='euclidean', device=device)
                sse = eval_kmeans(data, cluster_index, cluster_centers, device)
                print(str(cluster_centers.shape[0]) + " centers, sse:" + "{:.4f}".format(sse))
                sse_temp[i_loop] = sse
            sse_all.append(sse_temp.mean())
            sse_temp = np.zeros(attempt_kmeans)
    print(str([round(num, 3) for num in sse_all]).replace(".", ","))


if __name__ == '__main__':
    main()