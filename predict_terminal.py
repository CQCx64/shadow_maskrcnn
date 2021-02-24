import gc
import multiprocessing
import time

import torch
import torchvision
import get_net
import loadARP, loadMaskRcnn
import cv2
import os
from mxnet import image
import numpy as np

import utils

ARP_MODEL_NAME = 'res34_cbam_parallel'
IMG_PATH = 'images/strasbourg_000000_022067_leftImg8bit.png'
ARP_PATH = 'param/res34_bcam_parallel_625_0.2043_0.945_9.74.params'
SHADOW_PERCENT = 0.5

NUM_CLASS = 5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
RCNN_MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=NUM_CLASS)
CTX = utils.try_all_gpus()


def ARP_predict(return_dict, img_path=IMG_PATH, model_name=ARP_MODEL_NAME, arp_path=ARP_PATH,
                shadow_percent=SHADOW_PERCENT, mode='photo', write_path=''):
    net = get_net.get_net(model=model_name)
    net = loadARP.load_model(net, arp_path)
    if mode == 'photo':
        if isinstance(img_path, str):
            if os.path.exists(img_path):
                img = cv2.cvtColor(image.imread(img_path).asnumpy(), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
        pred = loadARP.predict(img, net).asnumpy()
        pred = np.array(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
        merge_img = loadARP.generate_shadow_mask(pred, img, shadow_percent)
        return_dict[0] = merge_img
    elif mode == 'video':
        video_capture = cv2.VideoCapture(img_path)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_shadow_writer = cv2.VideoWriter(write_path,
                                                   cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                                   fps, size)
        video_frame_num = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for num in range(video_frame_num):
            print(str(num) + '/' + str(video_frame_num))
            _, frame = video_capture.read()
            pred = loadARP.predict(frame, net).asnumpy()
            pred = np.array(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
            merge_img = loadARP.generate_shadow_mask(pred, frame, shadow_percent)
            video_shadow_writer.write(merge_img)
        video_shadow_writer.release()
        video_capture.release()
    print('done')


def rcnn_predict(return_dict, img, show_cv=True, model_path='param/model_new.pth', mode='photo', write_path=''):
    RCNN_MODEL.to(DEVICE)
    RCNN_MODEL.eval()
    model = loadMaskRcnn.load_model(RCNN_MODEL, model_path)
    if mode == 'photo':
        return_dict[0] = loadMaskRcnn.predict(img, model, mode='matrix', show_cv=show_cv)
    elif mode == 'video':
        video_capture = cv2.VideoCapture(img)
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_rcnn_writer = cv2.VideoWriter(write_path,
                                              cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                              fps, size)
        video_frame_num = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for num in range(video_frame_num):
            print(str(num) + '/' + str(video_frame_num))
            _, frame = video_capture.read()
            merge_img = loadMaskRcnn.predict(frame, model, mode='matrix', show_cv=show_cv)
            video_rcnn_writer.write(merge_img)
        video_rcnn_writer.release()
        video_capture.release()
    print('done')


if __name__ == '__main__':
    imgPath = r'D:\python\projects\shadow_maskrcnn\images\strasbourg_000000_004951_leftImg8bit.png'

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=ARP_predict, args=(return_dict, imgPath))
    p.start()
    p.join()
    arp_result = return_dict.values()[0]

    rcnn_predict(return_dict, arp_result)