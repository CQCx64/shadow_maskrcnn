from mxnet.gluon import nn


class CAM(nn.HybridBlock):
    def __init__(self, num_channels, ratio, **kwargs):
        super(CAM, self).__init__(**kwargs)
        with self.name_scope():
            self.avg_pool = nn.GlobalAvgPool2D()
            self.max_pool = nn.GlobalMaxPool2D()
            self.conv1 = nn.Conv2D(num_channels // ratio, 1, use_bias=False)
            self.conv2 = nn.Conv2D(num_channels, 1, use_bias=False)

    def hybrid_forward(self, F, X):
        X_avg = self.avg_pool(X)
        X_avg = self.conv1(X_avg)
        X_avg = F.relu(X_avg)
        X_avg = self.conv2(X_avg)

        X_max = self.max_pool(X)
        X_max = self.conv1(X_max)
        X_max = F.relu(X_max)
        X_max = self.conv2(X_max)

        Y = X_avg + X_max
        Y = F.sigmoid(Y)
        return Y


class SAM(nn.HybridBlock):
    def __init__(self, kernel_size=7, **kwargs):
        super(SAM, self).__init__(**kwargs)
        with self.name_scope():
            self.kernel_size = kernel_size
            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            self.padding = 3 if self.kernel_size == 7 else 1

            self.conv = nn.Conv2D(1, kernel_size=self.kernel_size, padding=self.padding, use_bias=False)

    def hybrid_forward(self, F, X):
        X_avg = F.mean(X, axis=1, keepdims=True)
        X_max = F.max(X, axis=1, keepdims=True)
        Y = F.concat(X_avg, X_max, dim=1)
        Y = self.conv(Y)
        Y = F.sigmoid(Y)
        return Y


class BCAM(nn.HybridBlock):
    def __init__(self, num_channels, ratio, **kwargs):
        super(BCAM, self).__init__(**kwargs)
        with self.name_scope():
            self.num_channels = num_channels
            self.ratio = ratio
            self.conv1 = nn.Conv2D(self.num_channels, kernel_size=3, strides=1, padding=1, use_bias=False)
            self.bn1 = nn.BatchNorm()
            # self.conv2 = nn.Conv2D(self.num_channels, kernel_size=3, strides=1, padding=1, use_bias=False)
            # self.bn2 = nn.BatchNorm()
            self.cam = CAM(self.num_channels, self.ratio)
            self.sam = SAM()

    def hybrid_forward(self, F, X):
        # residual = X
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.broadcast_mul(self.cam(Y), Y)
        Y = F.broadcast_mul(self.sam(Y), Y)
        # Y = Y + residual
        Y = F.relu(Y)
        return Y


class CBAM_modify(nn.HybridBlock):
    def __init__(self, num_channels, ratio, **kwargs):
        super(CBAM_modify, self).__init__(**kwargs)
        with self.name_scope():
            self.num_channels = num_channels
            self.ratio = ratio
            # self.conv1 = nn.Conv2D(self.num_channels, kernel_size=3, strides=1, padding=1, use_bias=False)
            # self.bn1 = nn.BatchNorm()
            # self.conv2 = nn.Conv2D(self.num_channels, kernel_size=3, strides=1, padding=1, use_bias=False)
            # self.bn2 = nn.BatchNorm()
            self.cam = CAM(self.num_channels, self.ratio)
            self.sam = SAM()

    def hybrid_forward(self, F, X):
        # residual = X
        # Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.broadcast_mul(self.cam(X), X)
        Y = F.broadcast_mul(self.sam(Y), Y)
        # Y = Y + residual
        Y = F.relu(Y)
        return Y
