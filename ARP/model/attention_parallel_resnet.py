from mxnet.gluon import nn, model_zoo
from attention_module import CBAM_modify, BCAM


class Parallel(nn.HybridBlock):
    """
    并行模块
    """

    def __init__(self, ptype, num_channels, extend=True, **kwargs):
        super(Parallel, self).__init__(**kwargs)
        assert ptype in ('left', 'right'), 'wrong type!'
        self.extend = extend
        with self.name_scope():
            if ptype == 'left':
                self.conv1 = nn.Conv2D(num_channels[0], 1)
                self.bn1 = nn.BatchNorm()
                self.conv2 = nn.Conv2D(num_channels[0], kernel_size=3, strides=1, padding=1)
                self.bn2 = nn.BatchNorm()
                self.conv3 = nn.Conv2D(num_channels[1], kernel_size=3, strides=1, padding=1)
                self.bn3 = nn.BatchNorm()
                self.conv4 = nn.Conv2D(num_channels[2], kernel_size=3, strides=1, padding=1)
                self.bn4 = nn.BatchNorm()
                self.conv5 = nn.Conv2D(num_channels[2], kernel_size=5, strides=1, padding=2)
                self.bn5 = nn.BatchNorm()
                if extend:
                    self.maxpool = nn.MaxPool2D()
                    self.conv6 = nn.Conv2D(num_channels[2], kernel_size=1, strides=1)
                    self.bn6 = nn.BatchNorm()
            elif ptype == 'right':
                self.conv1 = nn.Conv2D(num_channels[0], kernel_size=3, strides=1, padding=1)
                self.bn1 = nn.BatchNorm()
                self.conv2 = nn.Conv2D(num_channels[0], kernel_size=5, strides=1, padding=2)
                self.bn2 = nn.BatchNorm()
                self.conv3 = nn.Conv2D(num_channels[1], kernel_size=5, strides=1, padding=2)
                self.bn3 = nn.BatchNorm()
                self.conv4 = nn.Conv2D(num_channels[2], kernel_size=5, strides=1, padding=2)
                self.bn4 = nn.BatchNorm()
                self.conv5 = nn.Conv2D(num_channels[0], 1)
                self.bn5 = nn.BatchNorm()
                if extend:
                    self.maxpool = nn.MaxPool2D()
                    self.conv6 = nn.Conv2D(num_channels[0], kernel_size=1, strides=1)
                    self.bn6 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        if self.extend:
            X = self.bn6(self.maxpool(self.conv6(X)))
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = F.relu(self.bn3(self.conv3(Y)))
        Y = F.relu(self.bn4(self.conv4(Y)))
        Y = self.bn5(self.conv5(Y))

        Y = Y + X
        Y = F.relu(Y)
        return Y


class res34_cbam(nn.HybridSequential):
    def __init__(self, **kwargs):
        super(res34_cbam, self).__init__(**kwargs)
        res34 = model_zoo.vision.resnet34_v2(pretrained=True).features
        with self.name_scope():
            self.preoperation = res34[0:5]
            self.residual1 = res34[5]
            self.cbam1 = CBAM_modify(64, 8)
            self.residual2 = res34[6]
            self.cbam2 = CBAM_modify(128, 8)
            self.residual3 = res34[7]
            self.cbam3 = CBAM_modify(256, 8)
            self.residual4 = res34[8]
            self.cbam4 = CBAM_modify(512, 8)
            self.bn1 = res34[9]
            self.conv1 = nn.Conv2D(64, kernel_size=1)
            # self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(2, kernel_size=1)
            # self.bn3 = nn.BatchNorm()
            self.upsample = nn.Conv2DTranspose(2, kernel_size=64, padding=16, strides=32)

    def hybrid_forward(self, F, X):
        Y = self.preoperation(X)
        Y = self.residual1(Y)
        Y = self.cbam1(Y)
        Y = self.residual2(Y)
        Y = self.cbam2(Y)
        Y = self.residual3(Y)
        Y = self.cbam3(Y)
        Y = self.residual4(Y)
        Y = self.cbam4(Y)
        Y = F.relu(self.bn1(Y))
        Y = F.relu((self.conv1(Y)))
        Y = F.relu((self.conv2(Y)))
        Y = self.upsample(Y)
        return Y


class res34_cbam_parallel(nn.HybridSequential):
    def __init__(self, **kwargs):
        super(res34_cbam_parallel, self).__init__(**kwargs)
        res34 = model_zoo.vision.resnet34_v2(pretrained=True).features
        with self.name_scope():
            self.preoperation = res34[0:5]
            self.residual1 = res34[5]
            self.parallel1_0 = Parallel('left', (256, 128, 64), False)
            self.parallel1_1 = Parallel('right', (64, 128, 256), False)
            self.bcam1 = BCAM(64, 8)

            self.conv1 = nn.Conv2D(128, kernel_size=3, strides=2, padding=1)
            self.residual2 = res34[6]
            self.parallel2_0 = Parallel('left', (512, 256, 128))
            self.parallel2_1 = Parallel('right', (128, 256, 512))
            self.bcam2 = BCAM(128, 8)

            self.conv2 = nn.Conv2D(256, kernel_size=3, strides=2, padding=1)
            self.residual3 = res34[7]
            self.parallel3_0 = Parallel('left', (1024, 512, 256))
            self.parallel3_1 = Parallel('right', (256, 512, 1024))
            self.bcam3 = BCAM(256, 8)

            self.conv3 = nn.Conv2D(512, kernel_size=3, strides=2, padding=1)
            self.residual4 = res34[8]
            self.parallel4_0 = Parallel('left', (2048, 1024, 512))
            self.parallel4_1 = Parallel('right', (512, 1024, 2048))
            self.bcam4 = BCAM(512, 8)

            self.bn1 = res34[9]
            self.conv4 = nn.Conv2D(64, kernel_size=1)
            # self.bn2 = nn.BatchNorm()
            self.conv5 = nn.Conv2D(2, kernel_size=1)
            self.bn3 = nn.BatchNorm()
            self.upsample = nn.Conv2DTranspose(2, kernel_size=64, padding=16, strides=32)

    def hybrid_forward(self, F, X):
        Y = self.preoperation(X)
        out0 = Y
        concat0 = Y
        out1_0 = self.parallel1_0(out0)
        out1_1 = self.residual1(out0)
        out1_2 = self.parallel1_1(out0)
        out1 = F.concat(out1_0, out1_1, out1_2, dim=1)
        out1 = self.bcam1(F.concat(out1, concat0, dim=1))

        concat1 = self.conv1(out1)
        out2_0 = self.parallel2_0(out1)
        out2_1 = self.residual2(out1)
        out2_2 = self.parallel2_1(out1)
        out2 = F.concat(out2_0, out2_1, out2_2, dim=1)
        out2 = self.bcam2(F.concat(out2, concat1, dim=1))

        concat2 = self.conv2(out2)
        out3_0 = self.parallel3_0(out2)
        out3_1 = self.residual3(out2)
        out3_2 = self.parallel3_1(out2)
        out3 = F.concat(out3_0, out3_1, out3_2, dim=1)
        out3 = self.bcam3(F.concat(out3, concat2, dim=1))

        concat3 = self.conv3(out3)
        out4_0 = self.parallel4_0(out3)
        out4_1 = self.residual4(out3)
        out4_2 = self.parallel4_1(out3)
        out4 = F.concat(out4_0, out4_1, out4_2, dim=1)
        out4 = self.bcam4(F.concat(out4, concat3, dim=1))

        Y = F.relu(self.bn1(out4))
        Y = F.relu((self.conv4(Y)))
        Y = F.relu((self.bn3(self.conv5(Y))))
        Y = self.upsample(Y)
        return Y
