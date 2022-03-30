import os
import time
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import network
import config
import pandas as pd


class CNN_classify_module(nn.Module):
    def __init__(self):
        super(CNN_classify_module, self).__init__()
        self.cnn_classify_net = network.CNN_net().to(device=config.device)
        self.criterion = nn.BCELoss().to(device=config.device)
        # self.criterion = nn.MSELoss().to(device=config.device)
        self.optimizer = torch.optim.SGD(params=self.cnn_classify_net.parameters(), lr=config.cnn_learning_rate,
                                         momentum=config.cnn_weight_decay)

    def train(self, x_train, y_train):
        self.cnn_classify_net.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // config.cnn_batch_size

        print('### Training... ###')
        for epoch in range(1, config.cnn_max_epoch + 1):
            start_time = time.time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            loss_ = 0

            if epoch == 1:
                config.cnn_learning_rate = 0.1
                self.optimizer = torch.optim.SGD(params=self.cnn_classify_net.parameters(), lr=config.cnn_learning_rate,
                                                 momentum=config.cnn_weight_decay)
            elif epoch == 50:
                config.cnn_learning_rate = 0.01
                self.optimizer = torch.optim.SGD(params=self.cnn_classify_net.parameters(), lr=config.cnn_learning_rate,
                                                 momentum=config.cnn_weight_decay)
            elif epoch == 80:
                config.cnn_learning_rate = 0.001
                self.optimizer = torch.optim.SGD(params=self.cnn_classify_net.parameters(), lr=config.cnn_learning_rate,
                                                 momentum=config.cnn_weight_decay)

            for i in range(num_batches):
                # position = np.random.choice(range(num_samples), config.cnn_batch_size, replace=False)
                curr_batch_x_train = torch.tensor(
                    curr_x_train[config.cnn_batch_size * i:config.cnn_batch_size * (i + 1), :, :, :],
                    device=config.device)
                curr_batch_y_train = torch.tensor(
                    np.reshape(curr_y_train[config.cnn_batch_size * i:config.cnn_batch_size * (i + 1)], (-1, 1)),
                    device=config.device)

                # inputs, labels = data[0].to(config.device), data[1].to(config.device)

                curr_batch_y_pred = self.cnn_classify_net(curr_batch_x_train.double())

                loss = self.criterion(curr_batch_y_pred, curr_batch_y_train)
                loss_ += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss_ / num_batches, duration))

            if epoch % config.cnn_save_interval == 0:
                self.save(epoch)

    def test_or_validate(self, x, y, checkpoint_num_list):
        self.cnn_classify_net.eval()
        print('### Test or Validation ###')

        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(config.cnn_modeldir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)

            preds = []
            for i in tqdm(range(x.shape[0])):
                inter = x[i]
                inter = np.expand_dims(inter, axis=0)
                out_1 = self.cnn_classify_net(torch.tensor(inter, device=config.device))
                out = torch.round(out_1)
                preds.append(out.cpu().detach().numpy())

            # y = torch.tensor(y, device=config.device, dtype=torch.long)
            preds = np.reshape(preds, (-1,))
            sum = 0
            for i in range(y.shape[0]):
                if preds[i] == (y[i]):
                    sum += 1

            accuracy = sum / y.shape[0]
            print('Test accuracy: {:.4f}'.format(accuracy))
            # print('Test accuracy: {:.4f}'.format(torch.sum(torch.tensor(preds==y))/y.shape[0]))

    def save_weights(self, checkpoint_num_list):
        self.FCN_classify_net.eval()
        print('### Save Weight Matrices ###')

        isExist = os.path.exists(config.fcn_weights_savedir)
        if not isExist:
            os.makedirs(config.fcn_weights_savedir)
            print("The new directory is created")

        for checkpoint_num in checkpoint_num_list:
            count = 0
            for params in self.FCN_classify_net.state_dict():
                if count % 2 == 0:
                    weight = self.FCN_classify_net.state_dict()[params].cpu().numpy()
                    checkpoint_file = os.path.join(config.fcn_weights_savedir,
                                                   'model-%d-%s.csv' % (checkpoint_num, params.replace(".", "_")))
                    pd.DataFrame(weight).to_csv(checkpoint_file, header=False, index=False)
                elif count % 2 == 1:
                    bias = self.FCN_classify_net.state_dict()[params].cpu().numpy()
                    checkpoint_file = os.path.join(config.fcn_weights_savedir,
                                                   'model-%d-%s.csv' % (checkpoint_num, params.replace(".", "_")))
                    pd.DataFrame(bias).to_csv(checkpoint_file, header=False, index=False)

    def save(self, epoch):
        checkpoint_path = os.path.join(config.cnn_modeldir, 'model-%d.ckpt' % (epoch))
        if not os.path.exists(config.cnn_modeldir):
            os.makedirs(config.cnn_modeldir, exist_ok=True)
        torch.save(self.cnn_classify_net.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location=config.device)
        self.cnn_classify_net.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


class FCN_classify_module(nn.Module):
    def __init__(self):
        super(FCN_classify_module, self).__init__()
        self.FCN_classify_net = network.FCN_net().to(device=config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=self.FCN_classify_net.parameters(), lr=config.fcn_learning_rate,
                                         momentum=config.fcn_weight_decay)
        self.test = []

    def train(self, x_train, y_train):
        self.FCN_classify_net.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // config.fcn_batch_size

        print('### Training... ###')
        for epoch in range(1, config.fcn_max_epoch + 1):
            start_time = time.time()
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            if epoch == 1:
                config.fcn_learning_rate = 0.1
                self.optimizer = torch.optim.SGD(params=self.FCN_classify_net.parameters(),
                                                 lr=config.fcn_learning_rate, momentum=config.fcn_weight_decay)
            elif epoch == 300:
                config.fcn_learning_rate = 0.01
                self.optimizer = torch.optim.SGD(params=self.FCN_classify_net.parameters(),
                                                 lr=config.fcn_learning_rate, momentum=config.fcn_weight_decay)
            elif epoch == 800:
                config.fcn_learning_rate = 0.005
                self.optimizer = torch.optim.SGD(params=self.FCN_classify_net.parameters(),
                                                 lr=config.fcn_learning_rate, momentum=config.fcn_weight_decay)

            for i in range(num_batches):
                # position = np.random.choice(range(num_samples), config.cnn_batch_size, replace=False)
                curr_batch_x_train = torch.tensor(
                    np.reshape(curr_x_train[config.fcn_batch_size * i:config.fcn_batch_size * (i + 1), :, :, :],
                               (config.fcn_batch_size, -1)),
                    device=config.device)
                curr_batch_y_train = torch.tensor(
                    curr_y_train[config.fcn_batch_size * i:config.fcn_batch_size * (i + 1), :], device=config.device)

                curr_batch_y_pred = self.FCN_classify_net(curr_batch_x_train.float())

                loss = self.criterion(curr_batch_y_pred, curr_batch_y_train)
                loss_ = loss
                l1_lambda = 0.05
                # l2_norm = sum(p.pow(2.0).sum() for p in self.FCN_classify_net.parameters())
                # l1_norm = sum(p.abs().sum() for p in self.FCN_classify_net.parameters())
                # loss = loss + l1_lambda * l1_norm

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss_), end='\r', flush=True)
                # self.test.append(self.FCN_classify_net.model[0].weight.clone())
                # if i > 0:
                # print(torch.equal(self.test[i], self.test[i-1]))
                # print(sum(sum(self.test[i]-self.test[i-1])))

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss_, duration))

            if epoch % config.fcn_save_interval == 0:
                self.save(epoch)

    def test_or_validate(self, x, y, checkpoint_num_list):
        self.FCN_classify_net.eval()
        print('### Test or Validation ###')

        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(config.fcn_modeldir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)

            preds = []
            for i in tqdm(range(x.shape[0])):
                inter = x[i]
                inter = [inter]
                out_1 = self.FCN_classify_net(torch.tensor(inter, device=config.device))
                out = torch.argmax(out_1, dim=1)
                preds.append(out.cpu().numpy())

            # y = torch.tensor(y, device=config.device, dtype=torch.long)
            preds = np.reshape(preds, (-1,))
            sum = 0
            for i in range(y.shape[0]):
                if preds[i] == y[i]:
                    sum += 1

            accuracy = sum / y.shape[0]
            print('Test accuracy: {:.4f}'.format(accuracy))
            # print('Test accuracy: {:.4f}'.format(torch.sum(torch.tensor(preds==y))/y.shape[0]))

    def save_weights(self, checkpoint_num_list):
        self.FCN_classify_net.eval()
        print('### Save Weight Matrices ###')

        isExist = os.path.exists(config.fcn_weights_savedir)
        if not isExist:
            os.makedirs(config.fcn_weights_savedir)
            print("The new directory is created")

        for checkpoint_num in checkpoint_num_list:
            count = 0
            checkpoint_file = os.path.join(config.fcn_modeldir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)
            for params in self.FCN_classify_net.state_dict():
                if count % 2 == 0:
                    weight = self.FCN_classify_net.state_dict()[params].cpu().numpy()
                    checkpoint_file = os.path.join(config.fcn_weights_savedir,
                                                   'model-%d-%s.csv' % (checkpoint_num, params.replace(".", "_")))
                    pd.DataFrame(weight).to_csv(checkpoint_file, header=False, index=False)
                elif count % 2 == 1:
                    bias = self.FCN_classify_net.state_dict()[params].cpu().numpy()
                    checkpoint_file = os.path.join(config.fcn_weights_savedir,
                                                   'model-%d-%s.csv' % (checkpoint_num, params.replace(".", "_")))
                    pd.DataFrame(bias).to_csv(checkpoint_file, header=False, index=False)
                count += 1

    def save(self, epoch):
        checkpoint_path = os.path.join(config.fcn_modeldir, 'model-%d.ckpt' % (epoch))
        if not os.path.exists(config.fcn_modeldir):
            os.makedirs(config.fcn_modeldir, exist_ok=True)
        torch.save(self.FCN_classify_net.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location=config.device)
        self.FCN_classify_net.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
