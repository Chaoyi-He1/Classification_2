import data_read
import config
import model
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms


def main():
    print("--- Preparing Data ---")
    # training_input, training_type_one_hot, training_type = data_read.training_read(config.training_data_path)
    testing_input, testing_type_one_hot, testing_type = data_read.testing_read(config.testing_data_path)
    # validating_input, validating_type_one_hot, validating_type = data_read.validating_read(config.validate_data_path)

    cnn_model = model.CNN_classify_module()
    # fcn_model = model.FCN_classify_module()

    # cnn_model.train(training_input, training_type)
    cnn_model.test_or_validate(testing_input, testing_type, [60])
    # fcn_model.train(training_input, training_type_one_hot)
    # fcn_model.test_or_validate(training_input, training_type, [800])
    # fcn_model.save_weights([800])


def main_pic():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='../Pictest/data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='../Pictest/data', train=False, download=False, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model_train = model.CNN_classify_module()
    model_train.train(trainloader)


if __name__ == '__main__':
    main()
