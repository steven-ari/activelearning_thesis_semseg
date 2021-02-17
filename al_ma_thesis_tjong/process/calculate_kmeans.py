import os
from os.path import dirname as dr, abspath

from sklearn import preprocessing
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader

import al_ma_thesis_tjong.presets.dataset_preset as datasets_preset


def main():
    cluster_range = range(0, 21, 1)

    print("Start k-means clustering!")

    # plot kmeans distribution, coded MNIST
    mnist_root = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'Dataset_MNIST_n')
    # dataset, coded MNIST
    data_train, target_train = datasets_preset.provide_reduced_mnist(train=True)

    normalized_vectors = preprocessing.normalize(data_train)
    scores = [KMeans(n_clusters=i+2).fit(normalized_vectors).inertia_
              for i in cluster_range]
    print("Error scores for coded MNIST:")
    print(cluster_range)
    print(*[str(a).replace(".", ",") for a in scores], sep="\n")

    # dataset, unreduced MNIST
    data_train, target_train = datasets_preset.provide_unreduced_mnist(train=True)

    normalized_vectors = preprocessing.normalize(data_train)
    scores = [KMeans(n_clusters=i + 2).fit(normalized_vectors).inertia_
              for i in cluster_range]
    print("Error scores for unreduced MNIST:")
    print(cluster_range)
    print(*[str(a).replace(".", ",") for a in scores], sep="\n")

    # dataset, coded F-MNIST
    data_train, target_train = datasets_preset.provide_reduced_f_mnist(train=True)

    normalized_vectors = preprocessing.normalize(data_train)
    scores = [KMeans(n_clusters=i + 2).fit(normalized_vectors).inertia_
              for i in cluster_range]
    print("Error scores for coded F-MNIST:")
    print(cluster_range)
    print(*[str(a).replace(".", ",") for a in scores], sep="\n")

    # dataset, unreduced F-MNIST
    data_train, target_train = datasets_preset.provide_unreduced_f_mnist(train=True)

    normalized_vectors = preprocessing.normalize(data_train)
    scores = [KMeans(n_clusters=i + 2).fit(normalized_vectors).inertia_
              for i in cluster_range]
    print("Error scores for unreduced F-MNIST:")
    print(cluster_range)
    print(*[str(a).replace(".", ",") for a in scores], sep="\n")


if __name__ == '__main__':
    main()