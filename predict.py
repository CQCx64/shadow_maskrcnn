import get_net
import loadARP, loadMaskRcnn
import cv2
from mxnet import image
import numpy as np

ARP_MODEL_NAME = 'res34_cbam_parallel'
IMG_PATH = '../images/strasbourg_000000_004951_leftImg8bit.png'


def ARP_predict():
    img = cv2.cvtColor(image.imread(IMG_PATH).asnumpy(), cv2.COLOR_BGR2RGB)
    net = get_net.get_net(model=ARP_MODEL_NAME)
    net = loadARP.load_model(net)

    pred = loadARP.predict(IMG_PATH, net).asnumpy()
    pred = np.array(cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR), dtype=np.uint8)

    merge_img = loadARP.generate_shadow_mask(pred, img)
    return merge_img


if __name__ == '__main__':
    ARP_result = ARP_predict()
    cv2.imshow('sample', cv2.resize(ARP_result, (1024, 512)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()