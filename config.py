import torch
import numpy as np

training_data_path = './data/training_small.csv'
validate_data_path = './data/validate.csv'
testing_data_path = './data/testing.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sig_shape = [4, 64, 64]    # [2, 64, 64]
vec_size = 64*64*4  # 4096
type_position = int(vec_size)
spectrum_use_position = int(vec_size + 1)
num_classes = 1

## CNN parameters
kernel_size = [17, 9, 5, 3]
stride = [1, 1]
conv_net_width = [sig_shape[0], 16, 32, 32, 32]
pooling_shape = [16, 8, 4, 4]
cnn_fc_net_width = [int(conv_net_width[len(conv_net_width)-1] * vec_size / (4 * 2**(len(conv_net_width) - 1) * 2**(len(conv_net_width) - 1))), 256, num_classes]     #  int(conv_net_width[len(conv_net_width)-1] * vec_size * (2+len(conv_net_width)-1) / 1)
cnn_batch_size = 256
cnn_max_epoch = 100
cnn_weight_decay = 0.85
cnn_learning_rate = 0.1
cnn_save_interval = 10
cnn_modeldir = './Model/CNN_Model'

## Linear Network parameters

fcn_net_width = [vec_size * 2, 256, 256, 128, num_classes]
fcn_batch_size = 128
fcn_max_epoch = 1200
fcn_weight_decay = 0.1
fcn_learning_rate = 0.8
fcn_save_interval = 10
fcn_modeldir = './Model/FCN_Model'
fcn_weights_savedir = fcn_modeldir + '/Weights_Matrices'
