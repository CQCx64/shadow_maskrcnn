import os
import random

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import train.transforms as T
import numpy as np

from train import utils
from train.dataset import CityscapesDataset
from train.engine import train_one_epoch, evaluate


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                               hidden_layer,
                               num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(Resize(256))
    return T.Compose(transforms)


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return b, g, r


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 5  # 需要修改种类
    # use our dataset and defined transformations
    dataset = CityscapesDataset('/content/train', get_transform(train=True), instance=True)
    dataset_test = CityscapesDataset('/content/val', get_transform(train=False), instance=True)
    # dataset = CityscapesDataset('/gdrive/My Drive/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed', get_transform(train=True))
    # dataset_test = CityscapesDataset('/gdrive/My Drive/TorchVision_Maskrcnn/Maskrcnn/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:80]) #训练集张数
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[80:])#测试集张数

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)  # batch_size

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # train from start
    # get the model using our helper function
    # model = get_model_instance_segmentation(num_classes)
    # move model to the right device
    # model.to(device)

    # continue train
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.to(device)
    save = torch.load('model_5class_instance_log.pth')
    model.load_state_dict(save['model'])

    lr_rate = 0.003071  # init 0.005

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr_rate,
                                momentum=0.9, weight_decay=0.0005)  # 学习率
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=8,
                                                   gamma=0.85)

    model_without_ddp = model
    # let's train it for 10 epochs
    num_epochs = 100  # 训练次数

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        utils.save_on_master({
            'model': model.state_dict()},
            os.path.join('./', 'model_5class_instance_log.pth'))

        # utils.save_on_master({
        # 'model': model_without_ddp.state_dict()},
        # os.path.join('./', 'model1.pth'))

        evaluate(model, data_loader_test, device=device)

        # utils.save_on_master({
    #         'model': model_without_ddp.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'lr_scheduler': lr_scheduler.state_dict()},
    #         os.path.join('./', 'model_{}.pth'.format(epoch)))

    # utils.save_on_master({
    # 'model': model.state_dict()},
    # os.path.join('./', 'model_new.pth'))

    # utils.save_on_master({
    # 'model': model_without_ddp.state_dict()},
    # os.path.join('./', 'model1.pth'))

    print("That's it!")