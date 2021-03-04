# -*- coding: utf-8 -*-

import os
import sys
import pdb
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch


def read_data(data_dir, target_dir):
    data = sio.loadmat(data_dir)['indian_pines_corrected']
    target = sio.loadmat(target_dir)['indian_pines_gt']
    data = data.transpose(2, 0, 1)
    data = normalize(data)
    return data, target


def get_masks(target, train_prop, val_prop, save_dir=None):
    assert train_prop + val_prop < 1
    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        train_num = int(round(len(idx) * train_prop))
        val_num = int(round(len(idx) * val_prop))

        np.random.shuffle(idx)
        train_idx = idx[:train_num]
        val_idx = idx[train_num:train_num + val_num]
        test_idx = idx[train_num + val_num:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        folder_name = 'train_' + str(train_prop) + '_val_' + str(val_prop)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'), {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'), {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'), {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


def load_masks(target, train_prop, val_prop, mask_dir):
    # mask_dir: './data'
    mask_fname = os.path.join(mask_dir, 'train_' + str(train_prop) + '_val_' + str(val_prop))
    if not os.path.exists(mask_fname) or os.listdir(mask_fname) is None:
        train_mask, val_mask, test_mask = get_masks(target, train_prop, val_prop, save_dir=mask_dir)

    else:
        train_mask = sio.loadmat(os.path.join(mask_fname, 'train_mask.mat'))['train_mask']
        val_mask = sio.loadmat(os.path.join(mask_fname, 'val_mask.mat'))['val_mask']
        test_mask = sio.loadmat(os.path.join(mask_fname, 'test_mask.mat'))['test_mask']

    return train_mask, val_mask, test_mask


def get_samples(data, target, mask, to_tensor=True):
    data = data*mask
    target = target*mask

    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T
    target = target.ravel()
    data = data[target != 0]
    target = target[target != 0] - 1

    if to_tensor:
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

    return data, target


def get_patch_samples(data, target, mask, patch_size=13, to_tensor=True):
    # padding data
    width = patch_size // 2
    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'constant')
    target = np.pad(target, ((width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')

    # get patches
    patch_target = target * mask
    patch_target = patch_target[patch_target != 0] - 1
    patch_data = np.zeros((patch_target.shape[0], data.shape[0], patch_size, patch_size))
    index = np.argwhere(mask == 1)
    for i, loc in enumerate(index):
        patch = data[:, loc[0] - width:loc[0] + width + 1, loc[1] - width:loc[1] + width + 1]
        patch_data[i, :, :, :] = patch

    # shuffle
    state = np.random.get_state()
    np.random.shuffle(patch_data)
    np.random.set_state(state)
    np.random.shuffle(patch_target)

    # convert data format
    if to_tensor:
        patch_data = torch.from_numpy(patch_data).float()
        patch_target = torch.from_numpy(patch_target).long()
        if torch.cuda.is_available():
            patch_data = patch_data.cuda()
            patch_target = patch_target.cuda()

    return patch_data, patch_target


def get_all_patches(data, patch_size, to_tensor=True):
    width = patch_size // 2
    mask = np.ones((data.shape[1], data.shape[2]))
    patch_data = np.zeros((data.shape[1] * data.shape[2], data.shape[0], patch_size, patch_size))

    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')
    index = np.argwhere(mask)
    for i, loc in enumerate(index):
        patch_data[i, :, :, :] = data[:, loc[0] - width:loc[0] + width + 1, loc[1] - width:loc[1] + width + 1]

    if to_tensor:
        patch_data = torch.from_numpy(patch_data).float()
        if torch.cuda.is_available():
            patch_data = patch_data.cuda()

    return patch_data


def get_data(data, target, data_type, train_prop, val_prop, mask_dir, **kwargs):
    # data_type: 'patch', 'vector', 'full_image'
    # mask_dir: './data'

    # get masks
    train_mask, val_mask, test_mask = load_masks(target, train_prop, val_prop, mask_dir)

    # get data
    to_tensor = kwargs['to_tensor']
    if data_type == 'patch':
        try:
            patch_size = kwargs['patch_size']
        except:
            print("Get patch data, 'patch_size' not find!")
            sys.exit()
        train_data, train_target = get_patch_samples(data, target, train_mask, patch_size=patch_size, to_tensor=to_tensor)
        val_data, val_target = get_patch_samples(data, target, val_mask, patch_size=patch_size, to_tensor=to_tensor)
        test_data, test_target = get_patch_samples(data, target, test_mask, patch_size=patch_size, to_tensor=to_tensor)

    else:
        print("Get data, 'data_type' Error!")
        sys.exit()

    return train_data, train_target, val_data, val_target, test_data, test_target


def normalize(data):
    # data: channel*height*width
    data = data.astype(np.float)
    for i in range(len(data)):
        data[i, :, :] -= data[i, :, :].min()
        data[i, :, :] /= data[i, :, :].max()
    return data


def get_one_batch(train_data, train_target=None, batch_size=100):

    if train_target is None:
        train_target = torch.zeros(train_data.shape[0])
        train_target = torch.split(train_target, batch_size, dim=0)
    else:
        train_target = torch.split(train_target, batch_size, dim=0)

    train_data = torch.split(train_data, batch_size, dim=0)

    for i in range(len(train_data)):
        yield train_data[i], train_target[i]


def compute_accuracy(pred, target):
    accuracy = float((pred == target.data.cpu().numpy()).astype(int).sum()) / \
               float(target.size(0))  # compute accuracy
    return accuracy


def compute_accuracy_from_mask(pred, target, mask):
    # predict map: 145*145
    # target: ground truth 145*145
    # mask: one of train, validation and test masks
    pred = pred.copy()
    target = target.copy()
    # pred += 1
    pred = pred*mask
    target = target*mask

    pred = pred[pred != 0]
    target = target[target != 0]
    accuracy = float((pred == target).sum()) / float(len(pred))

    return accuracy


def plot_curves(loss, train_accuracy, test_accuracy):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 50))
    ax1.set_title('Loss', fontsize='x-large')
    ax2.set_title('Train and Test Accuracies', fontsize='x-large')
    ax1.plot(loss, color='r')
    ax2.plot(train_accuracy, color='r', label='Train Accuracy')
    ax2.plot(test_accuracy, color='g', label='Test Accuracy')
    legend = ax2.legend(fontsize='x-large', loc='lower right', shadow=True)
    #legend.get_frame().set_facecolor('C0')
    #plt.tight_layout()
    plt.show()


def plot_classification_maps(predict, target, **kwargs):
    predict_filt = predict.copy()
    predict_filt[target == 0] = 0
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    ax1.set_title('Ground truth')
    ax2.set_title('Predicted map')
    ax3.set_title('Remove background')
    ax1.imshow(target, **kwargs)
    ax2.imshow(predict, **kwargs)
    ax3.imshow(predict_filt, **kwargs)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.tight_layout()
    plt.show()


