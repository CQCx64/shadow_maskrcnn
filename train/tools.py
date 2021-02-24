import cv2
import numpy as np
from PIL import Image
import torch

names = {'0': 'backgroud', '1': 'person',  '2': 'rider', '3': 'car', '4': 'bus'}
color = {'0': (0, 0, 0), '1': (255, 0, 0), '2': (255, 255, 0), '3': (95, 158, 160), '4': (148, 0, 211)}


def transform_gt(path):
    img_origin = Image.open(path).resize((1024, 512), 0)
    img = np.array(img_origin)
    obj_ids = detect_label(img)
    boxes = draw_box(img, obj_ids)

    # 转变图片由实例到统一整体标签
    img = np.where(img // 1000 >= 1, img // 1000, img)
    img_b, img_g, img_r = img, img, img
    img_b = np.array(img_b)
    img_r = np.array(img_r)
    img_g = np.array(img_g)

    for i in color.keys():
        img_r[img_r == int(i)] = color.get(i)[0]
        img_g[img_g == int(i)] = color.get(i)[1]
        img_b[img_b == int(i)] = color.get(i)[2]
    img_rgb = img_origin.convert('RGB')
    img_rgb = np.array(img_rgb)
    img_rgb[:, :, 0] = img_b
    img_rgb[:, :, 1] = img_g
    img_rgb[:, :, 2] = img_r
    for box in boxes:
        cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), thickness=2)

    print('语义标签: ', np.unique(img))
    cv2.imshow('go', img_rgb)
    cv2.waitKey()


def detect_label(img):
    img_matrix = np.array(img)
    obj_ids = np.unique(img_matrix)[1:]
    print('实体id: ' + str(obj_ids))
    msg = '图片包含'
    msg += str(len(obj_ids)) + '个个体: '
    for i in obj_ids:
        idx = i//1000 if i//1000 >= 1 else i
        msg += str(i) + names.get(str(idx)) + ' '
    print(msg)
    return obj_ids


def draw_box(mask, obj_ids):
    masks = (mask == obj_ids[:, None, None])
    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
    print('boxes: ', boxes)
    obj_ids = np.where(obj_ids//1000 >= 1, obj_ids//1000, obj_ids)
    labels = torch.tensor(obj_ids, dtype=torch.int64)
    print('label: ', labels)
    return boxes


if __name__ == '__main__':
    gt_path = r'E:\BaiduNetdiskDownload\cityscapes_master\gtFine\train\aachen\aachen_000000_000019_gtFine_instanceTrainIds.png'
    transform_gt(gt_path)