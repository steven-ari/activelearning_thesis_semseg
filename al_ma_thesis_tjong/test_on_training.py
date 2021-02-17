import os
import csv
import statistics
import pickle
from os.path import dirname as dr, abspath
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import al_ma_thesis_tjong.presets.dataset_preset as datasets_preset
from al_ma_thesis_tjong.presets import qbc_preset as qbc_preset


def csv_train_reader(csv_path, data_name):

    title_list = []
    acc_list = []
    acc_avg_list = []
    acc_std_list = []
    batch_size_list = []
    committee_vote_list = []

    batch_size = []
    acc_test = []
    acc_avg = []
    acc_std = []
    committee_vote = []

    # open and iterate through .csv
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:  # first title
                title_list.append(row[0])
                line_count += 1
            elif row.__len__() == 1:  # title again
                title_list.append(row[0])
                acc_list.append(acc_test)  # append to bigger list
                acc_avg_list.append(np.asarray(acc_avg))
                acc_std_list.append(np.asarray(acc_std))
                batch_size_list.append(np.asarray(batch_size))
                committee_vote_list.append(np.asarray(committee_vote))
                acc_test = []
                acc_avg = []
                acc_std = []
                batch_size = []  # clean temporary list
                committee_vote = []
                line_count += 1
            else:  # acc numbers
                batch_size.append(int(row[0]))  # append to temp list
                acc_test.append(np.asarray([100*float(acc) for acc in row[1:-4]]))
                acc_avg.append(float(row[-2][:-1]))
                acc_std.append(float(row[-1][:-1]))
                committee_vote.append(float(row[-4][:-1]))
                line_count += 1
        acc_list.append(acc_test)
        acc_avg_list.append(np.asarray(acc_avg))
        acc_std_list.append(np.asarray(acc_std))
        batch_size_list.append(np.asarray(batch_size))
        committee_vote_list.append(np.asarray(committee_vote))

    csv_data = {
        "data_name": data_name,
        "title_list": title_list,
        "acc_list": acc_list,
        "acc_avg_list": acc_avg_list,
        "acc_std_list": acc_std_list,
        "batch_size_list": batch_size_list,
        "committee_vote_list": committee_vote_list,
    }
    return csv_data


def main():
    # load dataset
    datasets = []  # reduced_MNIST, reduced_F_MNIST, Dataset_MNIST_n, Dataset_F_MNIST_n

    # dataset, reduced mnist
    data_train_all, target_train_all = datasets_preset.provide_reduced_mnist(train=True)
    datasets.append((data_train_all, target_train_all))
    print('Reduced MNIST loaded')

    '''# dataset, unreduced mnist
    data_train_all, target_train_all = datasets_preset.provide_unreduced_mnist(train=True)
    datasets.append((data_train_all, target_train_all))
    print('Unreduced MNIST loaded')'''

    # dataset, reduced f-mnist
    data_train_all, target_train_all = datasets_preset.provide_reduced_f_mnist(train=True)
    datasets.append((data_train_all, target_train_all))
    print('Reduced F-MNIST loaded')

    '''# dataset, unreduced f-mnist
    data_train_all, target_train_all = datasets_preset.provide_unreduced_f_mnist(train=True)
    datasets.append((data_train_all, target_train_all))
    print('Unreduced F-MNIST loaded')'''

    # training run directory
    results_dir = [os.path.join(dr(dr(abspath(__file__))), 'results', 'reduced_mnist'),
                   os.path.join(dr(dr(abspath(__file__))), 'results', 'unreduced_mnist'),
                   os.path.join(dr(dr(abspath(__file__))), 'results', 'reduced_f_mnist'),
                   os.path.join(dr(dr(abspath(__file__))), 'results', 'unreduced_f_mnist')]

    results_dir = [os.path.join(dr(dr(abspath(__file__))), 'results', 'reduced_mnist'),
                   os.path.join(dr(dr(abspath(__file__))), 'results', 'reduced_f_mnist')]

    # loop through datasets
    for i_dataset in range(datasets.__len__()):
        models_dirs = [os.path.join(results_dir[i_dataset], run_dir)
                      for run_dir in sorted(os.listdir(results_dir[i_dataset])) if 'run_' in run_dir]
        models_dirs = models_dirs
        print(str(models_dirs))
        csv_path = os.path.join(results_dir[i_dataset], 'xgb_qbc.csv')
        csv_save_path = os.path.join(results_dir[i_dataset], 'xgb_qbc_train.csv')
        csv_data = csv_train_reader(csv_path, '')

        train_data = datasets[i_dataset][0]
        target_data = datasets[i_dataset][1]
        # loop through training runs
        for i_data_size in range(csv_data['batch_size_list'].__len__()):
            # to document training process
            train_text = [str(x) for x in csv_data['batch_size_list'][i_data_size]]
            run_path = [os.path.join(models_dirs[i_data_size], path)
                          for path in sorted(os.listdir(models_dirs[i_data_size])) if 'models_batch_' in path]
            # loop through batch in run, get committee
            for i_batch in range(run_path.__len__()):
                print('Load from: ' + run_path[i_batch])
                committee_list = pickle.load(open(run_path[i_batch], "rb"))
                output_list_train = np.zeros((committee_list.__len__(), train_data.shape[0]))
                # loop through committee, prediction for each model
                for i_model in range(committee_list.__len__()):
                    output_list_train[i_model, :] = committee_list[i_model].predict(train_data)
                # check committee vote
                acc_models = qbc_preset.each_model_acc(output_list_train, target_data)
                acc_committee = qbc_preset.committee_vote(output_list_train, target_data)  # committee vote method
                print("Batch " + str(i_batch) + ": average acc of models is " + "{:.3f}".format(acc_models.mean() * 100) + "%")
                print("Batch " + str(i_batch) + ": acc of committee is " + "{:.3f}".format(acc_committee * 100) + "%")

                # Document training progress
                train_text[i_batch] = train_text[i_batch] + ' '.join([";" + "{:.4f}".format(elem) for elem in acc_models])
                train_text[i_batch] = train_text[i_batch] + '; ' + "{:.3f}".format(acc_committee * 100) + '%'  # committee vote method

            # write text to csv
            title = [csv_data["title_list"][i_data_size]]
            with open(csv_save_path, mode='a+') as test_file:
                test_writer = csv.writer(test_file, delimiter=',')
                test_writer.writerow(title)
            # loop through train_text
            for i_text in range(0, train_text.__len__()):
                text = train_text[i_text].split(";")
                mean = statistics.mean([float(i) for i in text[1:-2]])
                var = statistics.variance([float(i) for i in text[1:-2]]) ** 0.5
                text.append("{:.3f}".format(mean * 100) + "%")
                text.append("{:.3f}".format(var * 100) + "%")
                with open(csv_save_path, mode='a+') as test_file:
                    test_writer = csv.writer(test_file, delimiter=';')
                    test_writer.writerow(text)
            print("Saved to csv! Data size: " + str(csv_data['batch_size_list'][i_data_size]) + ", Dataset:" +
                  str(i_dataset))


if __name__ == '__main__':
    main()