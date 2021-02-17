import os
from os.path import dirname as dr, abspath
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import namedtuple

from cityscapesscripts.helpers.labels import labels
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# define label
# labels_valid = [single_label for single_label in labels if not single_label.ignoreInEval]
labels_valid = labels
n_labels_valid = 20
ignored_color = (0, 0, 0)


def show_pred_result_single(fcn_model, index, dataset, device, criterion):
    '''

    :param fcn_model: fcn_model
    :param index: index of data within pytorch dataset
    :param dataset: pytorch dataset, to get the normalization
    :param device: where to make prediction, cuda or cpu
    :param criterion: loss function to calculate loss
    :return:
    '''
    # get colormap image
    colormap_path = os.path.join(dr(abspath(__file__)), 'colormap_cityscape.png')

    # make prediction
    fcn_model.to(device)
    fcn_model.eval()
    data_single, target_single = dataset.__getitem__(index)
    data, target = data_single.unsqueeze(dim=0).to(device), target_single.unsqueeze(dim=0).to(device)
    output = fcn_model(data)['out']
    output_single = output.detach().cpu()[0]
    del data, target, output, fcn_model

    plt.interactive(True)
    # get normalization on test dataset, must reverse it to plot the ori image
    transform = [x for x in dataset.transform.transforms if x.__class__.__name__.endswith('Normalize')][0]
    normalization = (transform.mean, transform.std)
    img_single, target_single = dataset.__getitem__(index)

    # convert data to images
    scene_img = unnormalize_img(img_single, normalization)
    target_img = target2color_single(target_single)
    colormap_img = Image.open(fp=colormap_path)
    output_img = output2color_single(output_single)
    iou = getIou(output_single.unsqueeze(dim=0), target_single.unsqueeze(dim=0))

    # plot it
    my_dpi = 100
    fig = plt.figure(figsize=(1800 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    ax_img = plt.subplot(gs[0])
    ax_target = plt.subplot(gs[1])
    ax_colormap = plt.subplot(gs[2])
    ax_output = plt.subplot(gs[3])

    ax_img.imshow(scene_img)
    ax_target.imshow(target_img)
    ax_colormap.imshow(colormap_img)
    ax_output.imshow(output_img)

    # make all look pretty
    ax_img.axis('off')
    ax_target.axis('off')
    ax_colormap.axis('off')
    ax_output.axis('off')
    fig_title = 'Prediction for data index: ' + str(index) + ', IOU: ' + "{:.3f}".format(iou[0])
    fig.suptitle(fig_title)
    plt.tight_layout()

    # save figure
    file_name = 'prediction_data_' + str(index)
    dir_name = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'prediction_examples')
    save_dir = os.path.join(dir_name, (file_name + '.png'))
    plt.savefig(save_dir, format='png', dpi=300)


def show_pred_result_single_coco(fcn_model, index, dataset, device, criterion):
    '''

    :param fcn_model: fcn_model
    :param index: index of data within pytorch dataset
    :param dataset: pytorch dataset, to get the normalization
    :param device: where to make prediction, cuda or cpu
    :param criterion: loss function to calculate loss
    :return:
    '''

    # make prediction
    fcn_model.to(device)
    fcn_model.eval()
    data_single, _ = dataset.__getitem__(index)
    data = data_single.unsqueeze(dim=0).to(device)
    output = fcn_model(data)['out']

    output_single = output.detach().cpu()[0]
    del data, fcn_model

    plt.interactive(True)
    # get normalization on test dataset, must reverse it to plot the ori image
    transform = [x for x in dataset.transform.transforms if x.__class__.__name__.endswith('Normalize')][0]
    normalization = (transform.mean, transform.std)
    img_single, target_single = dataset.__getitem__(index)

    # convert data to images
    scene_img = unnormalize_img(img_single, normalization)
    output_img = output2color_single(output_single)
    # iou = getIou(output_single.unsqueeze(dim=0), target_single.unsqueeze(dim=0))

    # plot it
    my_dpi = 100
    fig = plt.figure(figsize=(1800 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax_img = plt.subplot(gs[0])
    # ax_output_n_img = plt.subplot(gs[1])
    ax_output = plt.subplot(gs[1])

    ax_img.imshow(scene_img)
    ax_output.imshow(output_img)

    # make all look pretty
    ax_img.axis('off')
    # ax_output_n_img.axis('off')
    ax_output.axis('off')
    fig_title = 'Prediction for data index: ' + str(index)
    fig.suptitle(fig_title)
    plt.tight_layout()

    # save figure
    file_name = 'prediction_data_' + str(index)
    dir_name = os.path.join(dr(dr(dr(abspath(__file__)))), 'results', 'plots', 'prediction_examples_coco')
    save_dir = os.path.join(dir_name, (file_name + '.png'))
    plt.savefig(save_dir, format='png', dpi=300)


def label_id2label(seg_id):

    seg_id = seg_id.resize((800, 800), Image.NEAREST)

    id_to_trainId = {label.id: label.trainId for label in labels}
    id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

    label_img = id_to_trainId_map_func(seg_id)
    label_img = label_img

    return torch.from_numpy(label_img).long()


def label_id2label_700(seg_id):

    seg_id = seg_id.resize((700, 700), Image.NEAREST)

    id_to_trainId = {label.id: label.trainId for label in labels}
    id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

    label_img = id_to_trainId_map_func(seg_id)
    label_img = label_img

    return torch.from_numpy(label_img).long()


def color2label(seg_image):

    seg_image = seg_image.resize((512, 256), Image.NEAREST)
    seg_image = np.moveaxis(np.array(seg_image), -1, 0)
    seg_label = np.zeros((seg_image.shape[1], seg_image.shape[2])).astype(int)

    for i_label in range(labels_valid.__len__()):
        # create color array
        compare_clr = np.ones_like(seg_image)
        compare_clr[0, :, :] *= labels_valid[i_label].color[0]  # channel r
        compare_clr[1, :, :] *= labels_valid[i_label].color[1]  # channel g
        compare_clr[2, :, :] *= labels_valid[i_label].color[2]  # channel b
        compare_clr[3, :, :] = seg_image[3, :, :]  # channel b

        # compare with segmentation .png
        idx = np.all(seg_image == compare_clr, axis=0)
        seg_label[idx] = labels_valid[i_label].trainId

    return torch.from_numpy(seg_label).long()


def color2label_cropped(seg_image):

    # from (512, 256) to (512, 128)
    crop_left = 0
    crop_upper = 64
    crop_right = 512
    crop_lower = 64+128

    seg_image = seg_image.resize((512, 256), Image.NEAREST)
    seg_image = seg_image.crop((crop_left, crop_upper, crop_right, crop_lower))
    seg_image = np.moveaxis(np.array(seg_image), -1, 0)
    seg_label = np.zeros((seg_image.shape[1], seg_image.shape[2])).astype(int)

    for i_label in range(labels_valid.__len__()):
        # create color array
        compare_clr = np.ones_like(seg_image)
        compare_clr[0, :, :] *= labels_valid[i_label].color[0]  # channel r
        compare_clr[1, :, :] *= labels_valid[i_label].color[1]  # channel g
        compare_clr[2, :, :] *= labels_valid[i_label].color[2]  # channel b
        compare_clr[3, :, :] = seg_image[3, :, :]  # channel b

        # compare with segmentation .png
        idx = np.all(seg_image == compare_clr, axis=0)
        seg_label[idx] = labels_valid[i_label].trainId

    return torch.from_numpy(seg_label).long()


def unnormalize_img(image, normalization):
    '''
    :param image: input image should be in form C x H x W, with C = rgb, a single image as pytorch tensor
    :param normalization: example normalization: mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    :return: image: np array H x W x C, a single image
    '''
    image[0] = (image[0] * normalization[1][0]) + normalization[0][0]
    image[1] = (image[1] * normalization[1][1]) + normalization[0][1]
    image[2] = (image[2] * normalization[1][2]) + normalization[0][2]

    # return np.array() with HxWxC, can be used directly to plot
    if torch.is_tensor(image):  # into numpy
        image = image.detach().cpu().numpy()
    if image.shape[-1] != 3:  # from C x H x W to HxWxC
        image = np.moveaxis(image, 0, -1)

    return image


def target2color_single(target):
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

    image = np.moveaxis(image/255, 0, -1)

    return image


def output2color_single(output_single):
    '''

    :param output_single: n_label x H x W
    :return: image H x W x C, C is RGB
    '''

    output = output_single.detach().cpu().numpy()
    label = np.argmax(output, axis=0)

    image = target2color_single(label)

    return image


# convert batch output of fcn into numpy
def output2color_batch(output):

    image_batch = np.zeros((output.shape[0], output.shape[1], output.shape[2], 3))  # rgb batch

    # per data in batch
    for i_data in range(output.shape[0]):
        image_batch[i_data] = output2color_single(output[i_data])

    return image_batch


def getIou(outputs, targets):
    '''
    :param outputs: batch x label x H x W, tensor, could be cuda or cpu
    :param targets: batch x H x W, tensor, could be cuda or cpu
    :return: iou: batch, numpy vector
    '''

    # move to cpu
    if outputs.device.type == 'cuda':
        outputs_cpu = outputs.detach().cpu()
    else:
        outputs_cpu = outputs
    if targets.device.type == 'cuda':
        target_cpu = targets.detach().cpu()
    else:
        target_cpu = targets

    # make into numpy
    if torch.is_tensor(outputs_cpu):
        outputs_cpu = outputs_cpu.numpy()
    else:
        outputs_cpu = outputs_cpu
    if torch.is_tensor(target_cpu):
        target_cpu = target_cpu.numpy()
    else:
        target_cpu = target_cpu

    intersection_list = []
    union_list = []
    outputs_cpu = np.argmax(outputs_cpu, axis=1).astype(np.int8)  # BATCH x label x H x W => BATCH x H x W
    for i_label in range(n_labels_valid):
        output_label = outputs_cpu == i_label
        target_label = target_cpu == i_label
        intersection = (output_label & target_label).sum(axis=(1, 2))  # Will be zero if Truth=0 or Prediction=0, for each data in batch
        union = (output_label | target_label).sum(axis=(1, 2))  # Will be zero if both are 0, for each data in batch
        intersection_list.append(intersection)
        union_list.append(union)
    # Avoid 0/0, for each data in batch
    iou = (np.array(intersection_list).sum(axis=0) + 1e-10) / (np.array(union_list).sum(axis=0) + 1e-6)

    return iou


def getIou_committe(outputs, targets):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    iou_commitee = []
    for i_models in range(outputs.shape[0]):
        iou = getIou(outputs[i_models], targets)
        iou_commitee.append(iou)

    return np.array(iou_commitee)
