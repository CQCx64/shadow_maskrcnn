import os

import get_net
import utils
import cv2, numpy as np
from mxnet import nd, image

MODEL_NAME = 'res34_cbam_parallel'
CTX = utils.try_all_gpus()


def load_model(model, arp_path='../param/res34_bcam_parallel_625_0.2043_0.945_9.74.params'):
    net = model
    import_path = arp_path

    net.load_parameters(import_path)
    net.collect_params().reset_ctx(CTX)
    print('load {} finished on {}'.format(MODEL_NAME, CTX))

    return net


def predict(img, net):
    X = image.imread(img)

    X = normalize_image(X).as_in_context(CTX[0])
    # X = (X.astype('float32') / 255).as_in_context(CTX[0])
    X = X.transpose((2, 0, 1)).expand_dims(axis=0)
    pred = nd.argmin(net(X), axis=1)
    # pred = net(X) > 0.01
    return pred.reshape((X.shape[2], X.shape[3]))


def normalize_image(img):
    rgb_mean = nd.array([0.485, 0.456, 0.406])
    rgb_std = nd.array([0.229, 0.224, 0.225])
    return (img.astype('float32') / 255 - rgb_mean) / rgb_std


def generate_shadow_mask(pred, img, shadow_percent=0.5):
    merge_img = pred * img
    merge_img = cv2.addWeighted(img, 1, merge_img, shadow_percent, 0)

    return merge_img


if __name__ == '__main__':
    img_path = '../images/strasbourg_000000_004951_leftImg8bit.png'
    img = cv2.cvtColor(image.imread(img_path).asnumpy(), cv2.COLOR_BGR2RGB)
    net = get_net.get_net(model=MODEL_NAME)
    net = load_model(net)

    pred = predict(img_path, net).asnumpy()
    pred = np.array(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), dtype=np.uint8)

    merge_img = generate_shadow_mask(pred, img)
    # 保存
    # cv2.imwrite('sample.png', merge_img)

    cv2.imshow('sample', cv2.resize(merge_img, (1024, 512)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
