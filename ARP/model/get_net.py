import numpy as np
from mxnet import nd, init
from mxnet.gluon import nn

from attention_parallel_resnet import res34_cbam, res34_cbam_parallel


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)


def get_net(model='res34_cbam'):
    a = nd.random.uniform(shape=(1, 3, 64, 64))
    print(model)
    if model == 'res34_cbam':
        net = nn.HybridSequential()
        net = res34_cbam()

    elif model == 'res34_cbam_parallel':
        net = nn.HybridSequential()
        net = res34_cbam_parallel()

    print(model, ' loading')
    for layer in net:
        if net.prefix in layer.name and 'resnet' not in layer.name and layer != net[-1]:
            layer.initialize(init=init.Xavier(), force_reinit=True)
        elif layer == net[-1]:
            net[-1].initialize(init.Constant(bilinear_kernel(2, 2, 64)), force_reinit=True)
        # a = layer(a)

    print(model, ' initialize finished')

    return net
