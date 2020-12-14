import gc
import multiprocessing
import time

import torch
import torchvision
import get_net
import loadARP, loadMaskRcnn
import cv2
from mxnet import image
import numpy as np

ARP_MODEL_NAME = 'res34_cbam_parallel'
IMG_PATH = 'images/strasbourg_000000_022067_leftImg8bit.png'
ARP_PATH = 'param/res34_bcam_parallel_625_0.2043_0.945_9.74.params'
SHADOW_PERCENT = 0.5

NUM_CLASS = 10
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
RCNN_MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=NUM_CLASS)


def ARP_predict(return_dict, img_path=IMG_PATH, model_name=ARP_MODEL_NAME, arp_path=ARP_PATH, shadow_percent=SHADOW_PERCENT):
    img = cv2.cvtColor(image.imread(img_path).asnumpy(), cv2.COLOR_BGR2RGB)
    net = get_net.get_net(model=model_name)
    net = loadARP.load_model(net, arp_path)

    pred = loadARP.predict(img_path, net).asnumpy()
    pred = np.array(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), dtype=np.uint8)

    merge_img = loadARP.generate_shadow_mask(pred, img, shadow_percent)
    return_dict[0] = merge_img


def rcnn_predict(return_dict, img, show_cv=True, model_path='param/model_new.pth'):
    RCNN_MODEL.to(DEVICE)
    RCNN_MODEL.eval()
    model = loadMaskRcnn.load_model(RCNN_MODEL, model_path)
    return_dict[0] = loadMaskRcnn.predict(img, model, mode='matrix', show_cv=show_cv)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=ARP_predict, args=(return_dict, ))
    p.start()
    p.join()
    arp_result = return_dict.values()[0]

    rcnn_predict(arp_result)