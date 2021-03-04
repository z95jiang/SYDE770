import sys
from .basic_cnn import CNN

def get_net(net_type, input_shape, n_classes, config):

    if net_type == 'basic_cnn':
        assert len(input_shape) == 4
        input_shape = (1, ) + input_shape[1:]
        config = {'input_shape': input_shape,
                  'n_classes': n_classes,
                  'conv_layers': config['conv_layers'],
                  'feature_nums': config['feature_nums'],
                  'is_bn': config['is_bn'],
                  'conv_mode': config['conv_mode']
                  }
        net = basic_cnn.CNN(config)

    else:
        print('Net type not find!')
        sys.exit()

    return net












