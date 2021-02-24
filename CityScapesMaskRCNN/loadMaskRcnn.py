import random
import torch, torchvision
import cv2
import numpy as np

import cv2_util

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASS = 5


def load_model(model, param_path):
    params = torch.load(param_path)
    model.load_state_dict(params['model'])
    print('load maskrcnn complete')
    return model


def to_tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return b, g, r


def predict(image, model, mode='img', show_cv=False):
    img = cv2.imread(image) if mode == 'img' else image
    result = img.copy()
    dst = img.copy()
    img = to_tensor(img)

    # names = {'0': 'backgroud', '1': 'traffic light',
    #          '2': 'traffic sign', '3': 'person', '4': 'rider', '5': 'car',
    #          '6': 'truck', '7': 'bus', '8': 'motorcycle',
    #          '9': 'bicycle'}

    names = {'0': 'backgroud', '1': 'person', '2': 'rider', '3': 'car', '4': 'bus'}
    colors = {'0': (0, 0, 0), '1': (255, 0, 0), '2': (255, 255, 0), '3': (95, 158, 160), '4': (148, 0, 211)}
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(DEVICE)])
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    masks = prediction[0]['masks']

    m_bOK = False
    has_dst = False

    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.9:
            m_bOK = True
            color = colors.get(str(labels[idx].item()))
            mask = masks[idx, 0].mul(255).byte().cpu().numpy()
            thresh = mask
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # cv2.drawContours(dst, contours, -1, color, -1)

            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names.get(str(labels[idx].item()))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(result, text=' '+name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=color)

            dst1 = cv2.addWeighted(result, 0.7, dst, 0.3, 0)
            has_dst = True

    if m_bOK and show_cv and has_dst:
        cv2.imshow('sample', dst1)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if has_dst:
        return dst1
    else:
        return None


if __name__ == '__main__':
    img_path = '../images/strasbourg_000000_022067_leftImg8bit.png'

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=NUM_CLASS)
    model.to(DEVICE)
    model.eval()
    model = load_model(model, '../param/model_5class_instance.pth')
    predict(img_path, model, show_cv=True)
