import config
from torch import nn
import torch as F
import torch


class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.model = nn.ModuleList()
        for i in range(len(config.conv_net_width) - 1):
            self.model.append(nn.Conv2d(in_channels=config.conv_net_width[i], out_channels=config.conv_net_width[i+1],
                                        kernel_size=config.kernel_size[i], stride=config.stride,
                                        padding=int((config.kernel_size[i] - 1) / 2),
                                        dtype=float))   # , groups=config.conv_net_width[i]
            self.model.append(nn.BatchNorm2d(num_features=config.conv_net_width[i+1], dtype=float))
            self.model.append(nn.MaxPool2d(kernel_size=config.pooling_shape[i], stride=2,
                                           padding=int((config.pooling_shape[i] - 2) / 2)))

        for i in range(len(config.cnn_fc_net_width) - 1):
            self.model.append(nn.Linear(
                in_features=config.cnn_fc_net_width[i], out_features=config.cnn_fc_net_width[i+1], dtype=float))

    def forward(self, input_vec):
        output = input_vec
        for i in range(len(config.conv_net_width) - 1):
            output = self.model[3 * i](output)
            output = self.model[3 * i + 1](output)
            output = F.relu(output)
            output = self.model[3 * i + 2](output)

        # output = output.view(output.size(0), -1)
        output = torch.flatten(output, 1)

        for i in range(len(config.cnn_fc_net_width) - 1):
            output = self.model[i + int(3*(len(config.conv_net_width) - 1))](output)
            if i != len(config.cnn_fc_net_width) - 2:
                output = F.relu(output)
            else:
                output = F.sigmoid(output)
        # output = F.sigmoid(output)
        return output


class FCN_net(nn.Module):
    def __init__(self):
        super(FCN_net, self).__init__()
        self.model = nn.ModuleList()
        for i in range(len(config.fcn_net_width) - 1):
            self.model.append(nn.Linear(in_features=config.fcn_net_width[i], out_features=config.fcn_net_width[i+1]))

    def forward(self, input_vec):
        output = input_vec
        for i in range(len(config.fcn_net_width) - 1):
            output = self.model[i](output)
            if i != len(config.fcn_net_width) - 2:
                output = F.relu(output)
        return output
