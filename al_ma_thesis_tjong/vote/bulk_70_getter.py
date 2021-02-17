import os
from os.path import dirname as dr, abspath
import csv
import numpy as np
import random
from PIL import Image
import pickle
import math
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

# https://github.com/fregu856/deeplabv3


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


def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


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


def main(n_train, batch_train_size, n_test, batch_test_size):
    """
        :param n_model: number of models for the comittee
        :param n_train: number of training data to be used, this decides how long the training process will be
        :param batch_train_size: batch size for training process, keep it under 20
        :param idx_ratio: ratio of high entropy:ratio of random
        :return:
        """

    # paths
    img_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 've_test', 'example.png')
    save_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 've_test')
    csv_name_train = 'train.csv'
    csv_name_test = 'test.csv'
    csv_name_index = 'index.csv'
    dir_name = 'vote_bulk_70_from_90_005_'
    index_path_name = 'vote_90_5_70_005'
    save_weights_flag = True
    cityscape_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes')
    cityscape_loss_weight_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes', 'class_weights.pkl')
    cityscape_pretrain_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscape_pretrain')
    inference_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes', 'inference')
    color_path = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'cityscapes', 'color')
    print('cityscape_path: ' + cityscape_path)
    print(dir_name)
    print(index_path_name)

    # arguments
    n_train = 2880
    n_pretrain = 0
    n_test = 500
    n_epoch = 40
    test_factor = 3  # committee only tested every test_factor-th batch
    batch_train_size = 3*max(torch.cuda.device_count(), 1)
    batch_train_size_pretrain = 4
    batch_test_size = 25*max(torch.cuda.device_count(), 1)
    lr = 0.0001
    loss_print = 2
    idx_ratio = [0.0, 1.0]  # proportion to qbc:random
    continue_flag = False
    poly_exp = 1.0
    feature_extract = True
    manual_seed = 0
    np.random.seed(manual_seed)

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
                                                        target_type='semantic',
                                                        transform=transform,
                                                        target_transform=segmen_preset.label_id2label,
                                                        n=n_train)
    # read used index
    csv_path_index_source = os.path.join(save_path, index_path_name, csv_name_index)
    with open(csv_path_index_source) as csv_file:
        data = csv_file.readlines()
        train_index = np.array(list(map(int, data[-1][3:data[-1].find(';', (len(data[-1])-20))].split(','))))
        print(len(train_index))
        # np.random.shuffle(train_index)
        train_index = train_index[0:int(n_train*0.7)+1]
    print(len(train_index))
    train_dataset = Subset(train_dataset, indices=train_index)
    test_dataset = dataset_preset.Dataset_Cityscapes_n_i(root=cityscape_path, split='val', mode='fine',
                                                         target_type='semantic',
                                                         transform=transform,
                                                         target_transform=segmen_preset.label_id2label,
                                                         n=n_test)
    # only test on part of data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_train_size, shuffle=True,
                                  num_workers=3*max(torch.cuda.device_count(), 1), drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_test_size, shuffle=True,
                                 num_workers=3*max(torch.cuda.device_count(), 1), drop_last=True)
    print("Datasets loaded!")

    # create models, optimizers, scheduler, criterion
    # the models
    fcn_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True,
                                                                    num_classes=segmen_preset.n_labels_valid,
                                                                    aux_loss=True)
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model)

    # the optimizers
    params_to_update = fcn_model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in fcn_model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in fcn_model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    params = add_weight_decay(fcn_model, l2_value=0.0001)
    '''optimizer = torch.optim.SGD([{'params': fcn_model.module.classifier.parameters()},
                                  {'params': list(fcn_model.module.backbone.parameters()) +
                                             list(fcn_model.module.aux_classifier.parameters())}
                                  ], lr=lr, momentum=0.9)'''

    optimizer = torch.optim.Adam([{'params': fcn_model.module.classifier.parameters()},
                                  {'params': list(fcn_model.module.backbone.parameters()) +
                                             list(fcn_model.module.aux_classifier.parameters())}
                                  ], lr=lr, weight_decay=0.0001)
    lambda1 = lambda epoch: math.pow(1 - (epoch / n_epoch), poly_exp)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    with open(cityscape_loss_weight_path, "rb") as file:  # (needed for python3)
        class_weights = np.array(pickle.load(file))
    class_weights = torch.from_numpy(class_weights)
    class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()

    # report everything
    text = ('Model created' + (', n_train: ' + str(n_train)) + (', n_epoch: ' + str(n_epoch)) +
            (', batch_train_size: ' + str(batch_train_size)) + (', idx_ratio: ' + str(idx_ratio)) +
            (', n_test: ' + str(n_test)) + (', batch_test_size: ' + str(batch_test_size)) +
            (', test_factor: ' + str(test_factor)) + (', optimizer: ' + str(optimizer)) +
            (', model: ' + str(fcn_model)))
    print(text)

    # for documentation
    train_text = [str(x) for x in range(1, n_epoch+1)]
    test_text = [str(x) for x in range(1, n_epoch+1)]
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
    title = ["Training progress for n_model = " + str(1) + ", idx_ratio:  " + str(idx_ratio) +
             ', for multiple epoch, torch seed: ' + str(manual_seed)]
    with open(csv_path_train, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # write test progress
    csv_path_test = os.path.join(run_path, csv_name_test)
    title = ["Test progress for n_model = " + str(1) + ", idx_ratio:  " + str(idx_ratio)
             + ', for multiple epoch, torch seed: ' + str(manual_seed) + 'run_path: ' + run_path +
             'index_from: ' + index_path_name]
    with open(csv_path_test, mode='a+', newline='') as test_file:
        test_writer = csv.writer(test_file, delimiter=',')
        test_writer.writerow(title)

    # load from previous run if requested
    if continue_flag:
        fcn_model.load_state_dict(torch.load(
            'C:\\Users\\steve\\Desktop\\projects\\al_kitti\\results\\first_test\\adam_run_005\\model_weight_epoch_10.pt'))
        print('weight loaded')

    # training process, n-th batch
    for i_epoch in range(n_epoch):
        loss_epoch = []
        iou_epoch = []
        time_epoch = []
        for i_batch, (data_train, target_train) in enumerate(train_dataloader):

            t = Timer()
            t.start()
            # train batch
            output, loss, iou, fcn_model, optimizer = train_batch(fcn_model, data_train, target_train,
                                                                  optimizer, device, criterion)
            print('Epoch: ' + str(i_epoch) + '\t Batch: ' + str(i_batch) + '/' + str(len(train_dataloader))
                  + '; model ' + str(0) +
                  '; train loss avg: ' + "{:.3f}".format(loss) +
                  '; train iou avg: ' + "{:.3f}".format(iou.mean()))
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            loss_epoch.append(loss)
            iou_epoch.append(iou.mean())
            time_epoch.append(t.stop())

        # document train result
        train_text[i_epoch] = train_text[i_epoch] + ";{:.4f}".format(np.array(loss_epoch).mean()) + \
                              ";{:.4f}".format(np.array(iou_epoch).mean()) + \
                              ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr'])) + ';' + str(len(train_index))

        # update train documentation
        text = train_text[i_epoch].split(";")
        with open(csv_path_train, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        # one epoch ends here
        scheduler.step()
        print(optimizer)
        # save temporary model
        if i_epoch % 10 == 0 or (i_epoch+1) == n_epoch:
            fcn_model.train()
            torch.save(fcn_model.state_dict(), os.path.join(run_path, ('model_weight_epoch_train' +
                                                                       '{:03d}'.format(i_epoch) + '.pt')))
            fcn_model.eval()
            torch.save(fcn_model.state_dict(), os.path.join(run_path, ('model_weight_epoch_' +
                                                                       '{:03d}'.format(i_epoch) + '.pt')))

        # perform test
        create_pred_img(fcn_model, test_dataloader, inference_path, color_path)
        all_result_dict = cityscapes_eval()

        # average training time
        mean_time = np.array(time_epoch).mean()

        # document test result
        test_text[test_text_index] = test_text[test_text_index] + \
                                     ";{:.4f}".format(all_result_dict['averageScoreClasses']) + \
                                     ";{:.7f}".format(np.array(optimizer.param_groups[0]['lr'])) \
                                     + ";{:.4f}".format(mean_time) + ';' + str(len(train_index))

        # update test documentation
        text = test_text[test_text_index].split(";")
        with open(csv_path_test, mode='a+', newline='') as test_file:
            test_writer = csv.writer(test_file, delimiter=';')
            test_writer.writerow(text)

        test_text_index = test_text_index + 1


if __name__ == '__main__':
    main(2900, 4, 500, 50)
    # main(2970, 6, 500, 100)