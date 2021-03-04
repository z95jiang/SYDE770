# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
from copy import deepcopy
import torch
from torch import nn, optim
from tools import *
from model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


############################################# set super-parameters ############################################

TRAIN_PROP = 0.2
VAL_PROP = 0.2
BATCH_SIZE = 5000
PATCH_SIZE = 13
EPOCH = 5000
LR = 0.001
TEST_INTERVAL = 1
NET_TYPE = 'basic_cnn'  # 'bpnet', 'basic_cnn', 'resnet', 'dip_resnet'
DATA_TYPE = 'patch'  # 'patch'(resnet, cnn), 'vector'(bp), 'full_image'(dip_resnet)


CONV_LAYERS = 3
FEATURE_NUMS = [32, 64, 64]
IS_BN = True  # set 'True' means using batch normalization
CONV_MODE = 'same' 

config = dict(conv_layers=CONV_LAYERS, feature_nums=FEATURE_NUMS, is_bn=IS_BN, conv_mode=CONV_MODE)#, act_fun=ACT_FUN, pad=PAD)

#################################### prepare data and construct network #####################################

data_dir = './data/Indian_pines_corrected.mat'
target_dir = './data/Indian_pines_gt.mat'
mask_dir = './data'
data, target = read_data(data_dir, target_dir)

train_data, train_target, val_data, val_target, test_data, test_target = \
    get_data(data, target, DATA_TYPE, TRAIN_PROP, VAL_PROP, mask_dir, patch_size=PATCH_SIZE, to_tensor=True)
input_shape = train_data.shape
n_classes = train_target.max().item() + 1
model = get_net(NET_TYPE, input_shape, n_classes, config)

######################################### train model and save ###########################################
def train(model, train_data, train_target):

    global LR, EPOCH, BATCH_SIZE, NET_TYPE, TEST_INTERVAL, \
        val_data, val_target, test_data, test_target
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()  # loss function: cross entropy
    optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer: adam
    loss_list = []
    test_acc_list = []
    train_acc_list =[]
    best_test = 0
    save_dir = './model_save'
    state_dict = None
    best_state = None
    test_accuracy = None
    for epoch in range(EPOCH):

        for idx, samples in enumerate(get_one_batch(train_data, train_target, BATCH_SIZE)):
            data = samples[0]
            target = samples[1]
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % TEST_INTERVAL == 0:
                train_accuracy = test(model, train_data, train_target)[1]
                val_accuracy = test(model, val_data, val_target)[1]
                test_accuracy = test(model, test_data, test_target)[1]
                torch.cuda.empty_cache()
                print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: {3:.6f}　| Val: {4:.6f} | Test: {5:.6f}'.
                      format(epoch + 1, idx + 1, loss.item(), train_accuracy, val_accuracy, test_accuracy),
                      '\r', end='')
                if test_accuracy > best_test:
                    best_train = train_accuracy
                    best_val = val_accuracy
                    best_test = test_accuracy
                    best_state = [epoch + 1, idx + 1, loss, best_train, best_val, best_test]
                    state_dict = deepcopy(model.state_dict())
            loss_list.append(loss.item())
            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)

    plot_curves(loss_list, train_acc_list, test_acc_list)
    model_name = NET_TYPE + '_' + str(BATCH_SIZE) + '_' + str(EPOCH) + '.pkl'
    model_dir = os.path.join(save_dir, model_name)
    torch.save(state_dict, model_dir)
    print('Best Results: ')
    print('Epoch: {}  Batch: {}  Loss: {}  Train accuracy: {}  Val accuracy: {} Test accuracy: {}'.format(*best_state))



def test(model, data, target=None):

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    try:
        with torch.no_grad():
            output = model(data).cpu().data  # copy cuda tensor to host memory then convert to ndarray
    except:
        output = None
        for idx, batch_data in enumerate(get_one_batch(data, batch_size=2000)):
            with torch.no_grad():
                batch_output = model(batch_data[0]).cpu().data
            if idx == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), dim=0)
    pred = torch.max(output, dim=1)[1].numpy()

    accuracy = None
    if target is not None:
        target = target.cpu()
        accuracy = compute_accuracy(pred, target)

    return pred, accuracy


train(model, train_data, train_target)
pdb.set_trace()




