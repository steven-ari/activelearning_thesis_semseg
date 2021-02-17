import os
from os.path import dirname as dr, abspath
import csv
import numpy as np
import random
from PIL import Image
import math
import pickle
from scipy.stats import entropy
from shutil import copy

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, BatchSampler, SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

from cityscapesscripts.helpers.labels import labels

from al_ma_thesis_tjong.presets import segmen_preset as segmen_preset
from al_ma_thesis_tjong.presets import qbc_preset as qbc_preset
from al_ma_thesis_tjong.presets import dataset_preset as dataset_preset
from al_ma_thesis_tjong.presets import models_preset as models_preset
from al_ma_thesis_tjong.presets.evalPixelLevelSemanticLabeling import main as cityscapes_eval

from al_ma_thesis_tjong.process.timer import Timer


def create_pred_img(fcn_model, val_loader, inference_path, color_path):
    with torch.no_grad():
        # net.to(device)
        fcn_model.eval()
        """if torch.cuda.device_count() > 1:
            fcn_model = nn.DataParallel(fcn_model)"""
        for i_batch, (img, _, _, img_names) in enumerate(val_loader):
            img = img.cuda()
            print(img.shape)
            print(i_batch)
            with torch.no_grad():
                output = fcn_model(img)
                output_np = np.argmax(output['out'].detach().cpu().numpy(), axis=1)

            # converter from trainingID to labelID
            trainId_to_labelId = {label.trainId: label.id for label in labels}
            trainId_to_labelId[19] = 0
            trainId_to_labelId_func = np.vectorize(trainId_to_labelId.get)

            # converter from trainingID to color
            trainId2color = {label.trainId: label.color for label in labels}
            trainId2color[19] = (0, 0, 0)
            trainId2color_func = np.vectorize(trainId2color.get)

            # save every output as image file
            for i_img in range(output_np.shape[0]):
                # create label
                img_array = trainId_to_labelId_func(output_np[i_img]).astype(np.uint8)
                img = Image.fromarray(img_array).resize((2048, 1024), Image.NEAREST)
                file_name = os.path.join(inference_path, (img_names[i_img].replace('leftImg8bit', 'pred') + '.png'))
                img.save(os.path.join(inference_path, file_name), format='PNG')
                print('Saved: ' + str(os.path.join(inference_path, file_name)))

                # create color image prediction, original size
                img_array = target2color_single_pil(output_np[i_img])
                img = Image.fromarray(img_array)
                file_name = os.path.join(color_path, (img_names[i_img].replace('leftImg8bit', 'color_ori') + '.png'))
                img.save(os.path.join(color_path, file_name), format='PNG')
                print('Saved: ' + str(os.path.join(color_path, file_name)))

                # create color image prediction
                img_array = target2color_single_pil(output_np[i_img])
                img = Image.fromarray(img_array).resize((2048, 1024), Image.NEAREST)
                file_name = os.path.join(color_path, (img_names[i_img].replace('leftImg8bit', 'color') + '.png'))
                img.save(os.path.join(color_path, file_name), format='PNG')
                print('Saved: ' + str(os.path.join(color_path, file_name)))
                print(img_array.shape)
            del img, output


def target2color_single_pil(target):
    '''

    :param target: H x W, with label values
    :return: image H x W x C, C is RGB
    '''

    image = np.zeros((3, target.shape[0], target.shape[1]))  # rgb image

    trainId_to_color = {label.trainId: label.color for label in labels}
    trainId_to_color[19] = (0, 0, 0) # mark invalid as black
    trainId_to_color_func = np.vectorize(trainId_to_color.get)

    image[0] = trainId_to_color_func(target)[0]
    image[1] = trainId_to_color_func(target)[1]
    image[2] = trainId_to_color_func(target)[2]

    image = np.moveaxis(image, 0, -1).astype(np.uint8)
    return image


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
    loss_normal = criterion(output['out'], target)
    loss_aux = criterion(output['aux'], target)
    loss = loss_normal + 0.4*loss_aux
    iou = np.array([0, 0]) # segmen_preset.getIou(output['out'], target)
    loss.backward()
    optimizer.step()

    # tidy up
    output_cpu, loss_np = output['out'].to('cpu'), loss.item()
    fcn_model.eval()
    # fcn_model_cpu = fcn_model.to('cpu')
    del data, target, output, loss
    if device.type == "cuda":
        print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3))
              + ' Gb')  # show gpu usage
        print("Max Memory allocated:" + str(np.round(torch.cuda.max_memory_allocated(device) / 1e9, 3))
              + ' Gb')  # show gpu usage

    return output_cpu, loss_np, iou, fcn_model, optimizer


# using proper numpy implementation, this one is faster
def consensus_entropy_semseg(output_list):

    # output_list : n_models x n_data x n_label x H x W
    # output_norm : n_models x n_data x n_label x H x W
    # output_list_prob : n_data x n_label x H x W
    # en : n_data x H x W
    print(output_list.shape)
    output_norm = output_list / np.sum(output_list, axis=2)[:, :, None, :, :]  # classwise normalisation
    output_list_prob = np.sum(output_norm, axis=0) / output_norm.shape[0]  # avg prob for all models
    en = entropy(output_list_prob, base=2, axis=1)
    result = np.sum(en, axis=(-2, -1))/(output_list_prob.shape[-2]*output_list_prob.shape[-1])  # avg over entire pixels

    '''# To plot entropy with heatmap and colorbar
    fig = plt.figure()
    img = plt.imshow(np.squeeze(en))
    axes = fig.axes[0]
    cbar = axes.figure.colorbar(img, ax=axes)
    plt.title('Consensus Entropy with Dropout: 20%, img avg: ' + '{:03f}'.format(result[0]))
    plt.tight_layout()'''
    return result


def consensus_entropy_dropout(fcn_model, train_dataset, train_index, idx_ratio, batch_test_size, device, n_model, dropout_rate, i_epoch, n_data):
    sample_train, _, _, _ = train_dataset.__getitem__(1)
    ve_train_array = np.zeros(shape=train_dataset.__len__())
    fcn_model.eval()

    # no need to count entropy if training 100% random
    if idx_ratio[1] < 1.0:
        print('After enable: ')
        fcn_model, orig_prob_list = models_preset.enable_dropout(fcn_model, dropout_rate)

        # ve will be calculated only on data of this indices
        calc_ve_idx = np.array(range(train_dataset.__len__()))
        calc_ve_idx = calc_ve_idx[np.isin(calc_ve_idx, train_index, invert=True)]
        """# work only with half of data if the remaining training data is too large
        if calc_ve_idx.__len__() > (train_dataset.__len__() / 2):
            np.random.shuffle(calc_ve_idx)
            calc_ve_idx = calc_ve_idx[0:int(train_dataset.__len__() / 2)]"""
        print('CE of ' + str(calc_ve_idx.__len__()) + ' data')
        # create dataloader
        batch_sampler = BatchSampler(SubsetRandomSampler(calc_ve_idx), batch_size=batch_test_size, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
        print('CE Dataloader length: ' + str(len(train_dataloader)))
        softmax = nn.Softmax()
        with torch.no_grad():
            for i_batch, (data, target, index, _) in enumerate(train_dataloader):
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                if device.type == "cuda":  # show gpu usage
                    print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3)) + ' Gb')

                output_list_train = np.zeros((n_model, data.shape[0], segmen_preset.n_labels_valid,
                                              sample_train.shape[-2], sample_train.shape[-1]))
                # fcn_model.to(device)
                # get output of each model
                for i_model in range(n_model):
                    print('Model:' + str(i_model) + ', CE:' + str(i_batch) + '/' + str(len(train_dataloader))
                          + ', epoch: ' + str(i_epoch) + ', total CE: ' + str(len(calc_ve_idx)))
                    print('Shape data:' + str(data.shape))
                    output_list_train[i_model] = softmax(fcn_model(data)['out']).detach().cpu().numpy()
                    print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3)) + ' Gb')
                    print(output_list_train[i_model].mean())
                del data, target
                ve_batch = consensus_entropy_semseg(output_list_train)
                ve_train_array[index] = ve_batch

            if device.type == "cuda":  # show gpu usage
                print("Memory allocated:" + str(np.round(torch.cuda.memory_allocated(device) / 1e9, 3)) + ' Gb')

        fcn_model = models_preset.restore_dropout(fcn_model, orig_prob_list)
    # ve_train_array: array for easier processing, only one value for each data, which is average ve on every pixels
    ve_sorted = np.argsort(ve_train_array)[::-1]
    indices = ve_sorted[0:n_data]

    return indices, fcn_model


def main():
    """
    :param n_model: number of models for the comittee
    :param n_train: number of training data to be used, this decides how long the training process will be
    :param batch_train_size: batch size for training process, keep it under 20
    :param idx_ratio: ratio of high entropy:ratio of random
    :return:
    """

    # paths
    save_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'consensus_test')
    csv_name_train = 'train.csv'
    csv_name_test = 'test.csv'
    csv_name_index = 'index.csv'
    dir_name = 'consensus_90_5_70_'
    save_weights_flag = True
    cityscape_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes')
    cityscape_loss_weight_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes', 'class_weights.pkl')
    cityscape_pretrain_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscape_pretrain')
    inference_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes', 'inference')
    color_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes', 'color')
    print('cityscape_path: ' + cityscape_path)
    print(dir_name)

    # arguments
    n_train = 2880  # divisible by 8: batch size and 10: 10% increment of training data increase
    n_pretrain = 0
    n_test = 500
    n_epoch = 40
    n_model = 10
    test_factor = 3  # committee only tested every test_factor-th batch
    batch_train_size = 3*max(torch.cuda.device_count(), 1)
    batch_test_size = 20*max(torch.cuda.device_count(), 1)
    lr = 0.0001
    loss_print = 2
    continue_flag = False
    poly_exp = 1.0
    feature_extract = True
    dropout_rate = 0.9
    idx_ratio = [1.0, 0.0]
    data_limit = 0.7
    manual_seed = 1

    # report qbc semseg to user in terminal
    text = (('n_model(dropout): ' + str(n_model)) + (', n_train: ' + str(n_train)) +
            (', batch_train_size: ' + str(batch_train_size)) +
            (', idx_ratio: ' + str(idx_ratio)) + (', test_factor: ' + str(test_factor)))
    print(text)

    # CUDA
    cuda_flag = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_flag else "cpu")
    device_cpu = torch.device("cpu")
    dataloader_kwargs = {'pin_memory': True} if cuda_flag else {}
    print(torch.cuda.device_count(), "GPUs detected")
    torch.manual_seed(manual_seed)
    # print("Max memory allocated:" + str(np.round(torch.cuda.max_memory_allocated(device) / 1e9, 3)) + ' Gb')

    # get data and index library
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize((800, 800), Image.BICUBIC), T.ToTensor(), T.Normalize(*mean_std)])
    train_dataset = dataset_preset.Dataset_Cityscapes_n(root=cityscape_path, split='train', mode='fine',
                                                        target_type='semantic', transform=transform,
                                                        target_transform=segmen_preset.label_id2label,
                                                        n=n_train)
    train_dataset_idx = dataset_preset.Dataset_Cityscapes_n_i(root=cityscape_path, split='train', mode='fine',
                                                              target_type='semantic', transform=transform,
                                                              target_transform=segmen_preset.label_id2label,
                                                              n=n_train)  # also get index of data
    test_dataset = dataset_preset.Dataset_Cityscapes_n_i(root=cityscape_path, split='val', mode='fine',
                                                         target_type='semantic', transform=transform,
                                                         target_transform=segmen_preset.label_id2label, n=n_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_test_size, shuffle=True,
                                 num_workers=3*max(torch.cuda.device_count(), 1), drop_last=False)
    print("Datasets loaded!")

    # create models, optimizers, scheduler, criterion, the model
    fcn_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True,
                                                                    num_classes=segmen_preset.n_labels_valid,
                                                                    aux_loss=True)
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model)

    # the optimizers
    optimizer = torch.optim.Adam([{'params': fcn_model.module.classifier.parameters()},
                                  {'params': list(fcn_model.module.backbone.parameters()) +
                                             list(fcn_model.module.aux_classifier.parameters())}
                                  ], lr=lr)
    lambda1 = lambda epoch: math.pow(1 - (epoch / n_epoch), poly_exp)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    with open(cityscape_loss_weight_path, "rb") as file:  # (needed for python3)
        class_weights = np.array(pickle.load(file))
    class_weights = torch.from_numpy(class_weights)
    class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()

    # report everything
    print(str(n_model) + " fcn models created")
    text = ('n_model: ' + str(n_model)) + (', n_train: ' + str(n_train)) + (', n_epoch: ' + str(n_epoch)) +\
           (', batch_train_size: ' + str(batch_train_size)) + (', idx_ratio: ' + str(idx_ratio))
    print(text)

    # to document training process, create directory, etc
    train_text = [str(x) for x in range(1, n_epoch+1)]
    test_text = [str(x) for x in range(1, n_epoch+1)]
    train_index_text = [str(x) for x in range(1, 8)]
    train_index_docu = 0
    train_index = []
    test_text_index = 0

    # write text to csv
    dir_number = 1
    while os.path.exists(os.path.join(save_path, (dir_name + '{:03d}'.format(dir_number)))):
        dir_number += 1
    run_path = os.path.join(save_path, (dir_name + '{:03d}'.format(dir_number)))
    os.makedirs(run_path)  # make run_* dir
    f = open(os.path.join(run_path, 'info.txt'), 'w+')  # write .txt file
    f.write(text)
    f.close()
    copy(__file__, os.path.join(run_path, os.path.basename(__file__)))

    # write training progress
    csv_path_train = os.path.join(run_path, csv_name_train)
    title = ["Training progress for n_model = " + str(n_model) + ", idx_ratio:  " + str(idx_ratio) + ', for multiple epoch']
    with open(csv_path_train, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # write test progress
    csv_path_test = os.path.join(run_path, csv_name_test)
    title = ["Test progress for n_model = " + str(1) + ", idx_ratio:  " + str(idx_ratio)
             + ', for multiple epoch, torch seed: ' + str(manual_seed) + 'run_path: ' + run_path]
    with open(csv_path_test, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # write index and train progress
    csv_path_index = os.path.join(run_path, csv_name_index)
    title = ["Index progress for n_model = " + str(n_model) + ", idx_ratio:  " + str(idx_ratio) + ', for multiple epoch']
    with open(csv_path_train, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # training start
    for i_epoch in range(n_epoch):
        # initialize with random
        if len(train_index) == 0:
            train_index = np.array(random.sample(range(n_train), k=int(n_train/10)))
            train_index_text[train_index_docu] = train_index_text[train_index_docu] + ': ' \
                                                 + str([x for x in train_index]).strip('[]')

            # update train and index documentation
            text = train_index_text[train_index_docu].split(";")
            with open(csv_path_index, mode='a+', newline='') as test_file:
                test_writer = csv.writer(test_file, delimiter=';')
                test_writer.writerow(text)
            print(train_index_text)
            train_index_docu = train_index_docu + 1
        # append with vote entropy
        elif (len(train_index) < int(0.7*n_train)) and (i_epoch % 5 == 0):
            t = Timer()
            t.start()
            # perform vote entropy on entire dataset
            indices, fcn_model = consensus_entropy_dropout(fcn_model, train_dataset_idx, train_index, idx_ratio,
                                                           batch_test_size, device, n_model, dropout_rate, i_epoch,
                                                           n_data=int(n_train/10))
            train_index = np.append(train_index, indices)
            train_index_text[train_index_docu] = train_index_text[train_index_docu] + ': ' + \
                                                 str([x for x in train_index]).strip('[]') +\
                                                 ";{:.4f}".format(np.array(t.stop()).mean())

            # update train and index documentation
            text = train_index_text[train_index_docu].split(";")
            with open(csv_path_index, mode='a+', newline='') as test_file:
                test_writer = csv.writer(test_file, delimiter=';')
                test_writer.writerow(text)
            print(train_index_text)
            train_index_docu = train_index_docu + 1

        # retrain with selected data
        print(train_index)
        print('length: ' + str(len(train_index)))
        train_subset = Subset(train_dataset_idx, train_index)
        train_dataloader = DataLoader(train_subset, batch_size=batch_train_size, shuffle=True)
        loss_epoch = []
        time_epoch = []
        for i_batch, (data_train, target_train, index, _) in enumerate(train_dataloader):
            # train batch
            t = Timer()
            t.start()
            output, loss, iou, fcn_model, optimizer = train_batch(fcn_model, data_train, target_train,
                                                                  optimizer, device, criterion)
            print('Epoch: ' + str(i_epoch) + '\t Batch: ' + str(i_batch) + '/' + str(len(train_dataloader))
                  + '; model ' + str(0) +
                  '; train loss avg: ' + "{:.3f}".format(loss) +
                  '; train iou avg: ' + "{:.3f}".format(iou.mean()))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            loss_epoch.append(loss)
            time_epoch.append(t.stop())

        # document train result
        train_text[i_epoch] = train_text[i_epoch] + ";{:.4f}".format(np.array(loss_epoch).mean()) + \
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr']))\
                              + ';' + str(len(train_index)) + ";{:.4f}".format(np.array(time_epoch).mean())

        # update train documentation
        text = train_text[i_epoch].split(";")
        with open(csv_path_train, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        # save temporary model and perform test
        if i_epoch % 10 == 0 or (i_epoch+1) == n_epoch:
            print('Save and Test Model')
            fcn_model.train()
            torch.save(fcn_model.state_dict(), os.path.join(run_path, ('model_weight_epoch_train' +
                                                                       '{:03d}'.format(i_epoch) + '.pt')))
            fcn_model.eval()
            torch.save(fcn_model.state_dict(), os.path.join(run_path, ('model_weight_epoch_' +
                                                                       '{:03d}'.format(i_epoch) + '.pt')))

        # perform test
        test_idx = test_text_index
        create_pred_img(fcn_model, test_dataloader, inference_path, color_path)
        all_result_dict = cityscapes_eval()

        # document test result
        test_text[test_idx] = test_text[test_idx] + ";{:.4f}".format(all_result_dict['averageScoreClasses']) +\
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr']))\
                              + ';' + str(len(train_index))

        # update test documentation
        text = test_text[test_idx].split(";")
        with open(csv_path_test, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        test_text_index = test_text_index + 1

        # one epoch ends here
        scheduler.step()
        print(optimizer)


if __name__ == '__main__':
    main()  # qbc_semseg(n_model, n_train, batch_size)
