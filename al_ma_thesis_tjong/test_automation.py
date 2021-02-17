'''Automate test on entire MNIST and F_MNIST, both reduced and unreduced'''

from al_ma_thesis_tjong import qbc_xgboost as qbc_test


def main():
    qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[0.0, 0.0, 1.0], dataset='unreduced_mnist')

    for i in range(5):
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[0.0, 1.0, 0.0], dataset='unreduced_mnist')

    for i in range(10):
        # large
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[1.0, 0.0, 0.0], dataset='unreduced_mnist')

    for i in range(10):
        # large
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_f_mnist')

        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=1000, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_mnist')

    for i in range(10):
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_f_mnist')

        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=3000, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_mnist')

        '''# small
        qbc_test.qbc(n_model=20, n_train=500, batch_size=100, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=500, batch_size=100, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=500, batch_size=100, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_f_mnist')

        qbc_test.qbc(n_model=20, n_train=500, batch_size=100, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=500, batch_size=100, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=500, batch_size=100, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_mnist')

        # medium
        qbc_test.qbc(n_model=20, n_train=4000, batch_size=500, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=4000, batch_size=500, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=4000, batch_size=500, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_f_mnist')

        qbc_test.qbc(n_model=20, n_train=4000, batch_size=500, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=4000, batch_size=500, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=4000, batch_size=500, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_mnist')

        # large
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=5000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=5000, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_f_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=5000, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_f_mnist')

        qbc_test.qbc(n_model=20, n_train=60000, batch_size=5000, idx_ratio=[1.0, 0.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=5000, idx_ratio=[0.0, 1.0, 0.0], dataset='reduced_mnist')
        qbc_test.qbc(n_model=20, n_train=60000, batch_size=5000, idx_ratio=[0.0, 0.0, 1.0], dataset='reduced_mnist')'''


if __name__ == '__main__':
    main()