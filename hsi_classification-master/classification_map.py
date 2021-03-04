import numpy as np
import torch
from model import basic_cnn
from tools import *

def get_all_patches(data, patch_size):
    width = patch_size // 2
    mask = np.ones((data.shape[1], data.shape[2]))
    patch_data = np.zeros((data.shape[1] * data.shape[2], data.shape[0], patch_size, patch_size))
    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')
    index = np.argwhere(mask)
    for i, loc in enumerate(index):
        patch_data[i, :, :, :] = data[:, loc[0] - width:loc[0] + width + 1, loc[1] - width:loc[1] + width + 1]
    return patch_data

def test(model, data, target=None):
    model.eval()
    output = model(data)
    output = output.cpu()
    pred = torch.max(output, 1)[1].data.numpy()
    accuracy = None
    if target is not None:
        target = target.cpu()
        accuracy = compute_accuracy(pred, target)
    return pred, accuracy

data_dir = './data/Indian_pines_corrected.mat'
target_dir = './data/Indian_pines_gt.mat'
model_dir = './model_save/basic_cnn_5000_5000.pkl'

patch_size = 13
config = {'input_shape': (1, 200, 13, 13),
          'n_classes': 16,
          'conv_layers': 3,
          'conv_mode':'same',
          'feature_nums': [32, 64, 64],
          'is_bn': True
          }

data, target = read_data(data_dir, target_dir)
patch_data = get_all_patches(data, patch_size)
patch_data = torch.from_numpy(patch_data).float().cuda()

model = basic_cnn.CNN(config).cuda()
model.load_state_dict(torch.load(model_dir))
pred = test(model, patch_data)[0]
map = pred.reshape(145, 145)

plot_classification_maps(map, target, cmap='jet')





