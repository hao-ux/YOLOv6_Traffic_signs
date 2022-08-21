import warnings
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, Add, MaxPool2D, Concatenate
)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


def ConvBn(inputs, filters, kernel_size, stride, padding, groups=1, weight_decay=5e-4):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, 
               groups=groups, use_bias=False, kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    return x


# RepVGG网络中的卷积模块
def RepVGGBlock(inputs, filters, kernel_size=3, stride=1, padding='same', dilation=1, groups=1,
                deploy=False, weight_decay=5e-4):
    if deploy:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, 
                   dilation_rate=dilation, use_bias=True, kernel_initializer=RandomNormal(stddev=0.02),
                   kernel_regularizer=l2(weight_decay))(inputs)
        x = ReLU()(x)
        return x
    else:
        if inputs.shape[-1] == filters and stride == 1:
            x0 = ConvBn(inputs, filters, kernel_size, stride, padding=padding, groups=groups)
            x1 = ConvBn(inputs, filters, 1, stride, padding=padding, groups=groups)
            x = Add()([x0, x1, BatchNormalization()(inputs)])
            return ReLU()(x)
        else:
            x0 = ConvBn(inputs, filters, kernel_size, stride, padding=padding, groups=groups)
            x1 = ConvBn(inputs, filters, 1, stride, padding=padding, groups=groups)
            x = Add()([x0, x1])
            return ReLU()(x)

# RepBlock -> [RepConv x n]
def RepBlock(inputs, filters, n=1, block=RepVGGBlock, weight_decay=5e-4):
    x = block(inputs=inputs, filters=filters, weight_decay=weight_decay)
    if n > 1:
        for _ in range(n-1):
            x = block(x, filters=filters, weight_decay=weight_decay)
        return x
    else:
        return x

def SimConv(inputs, filters, kernel_size, stride, groups=1, bias=False, padding='same', weight_decay=5e-4):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
               groups=groups, use_bias=bias, kernel_initializer=RandomNormal(stddev=0.02),
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x) 
    x = ReLU()(x)
    return x

# 原始的 SPPF 优化设计
def SimSPPF(inputs, filters, kernel_size=5, weight_decay=5e-4):
    c_ = filters // 2
    x = SimConv(inputs, c_, 1, 1, padding='valid', weight_decay=weight_decay)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y1 = MaxPool2D(pool_size=kernel_size, strides=1, padding='same')(x)
        y2 = MaxPool2D(pool_size=kernel_size, strides=1, padding='same')(y1)
        y3 = MaxPool2D(pool_size=kernel_size, strides=1, padding='same')(y2)
        out = Concatenate(axis=-1)([x, y1, y2, y3])
        return SimConv(out, filters, 1, 1, padding='valid', weight_decay=weight_decay)
    
# EfficientRep Backbone
# 该 Backbone 能够高效利用硬件（如 GPU）算力的同时，还具有较强的表征能力。
def EfficientRep(inputs, channels_list=None, num_repeats=None, block=RepVGGBlock, weight_decay=5e-4):
    # 640, 640, 3 -> 320, 320, 32 ---------------- yolov6s
    stem = block(inputs, filters=channels_list[0], kernel_size=3, stride=2, weight_decay=weight_decay)

    # 320, 320, 32 -> 160, 160, 64 ---------------- yolov6s
    ERBlock_2 = block(stem, filters=channels_list[1], kernel_size=3, stride=2, weight_decay=weight_decay)
    ERBlock_2 = RepBlock(ERBlock_2, filters=channels_list[1], n=num_repeats[1], block=block, weight_decay=weight_decay)

    # 160, 160, 64 -> 80, 80, 128 ---------------- yolov6s
    ERBlock_3 = block(ERBlock_2, filters=channels_list[2], kernel_size=3, stride=2, weight_decay=weight_decay)
    ERBlock_3 = RepBlock(ERBlock_3, filters=channels_list[2], n=num_repeats[2], block=block, weight_decay=weight_decay)

    # 80, 80, 128 -> 40, 40, 256 ---------------- yolov6s
    ERBlock_4 = block(ERBlock_3, filters=channels_list[3], kernel_size=3, stride=2, weight_decay=weight_decay)
    ERBlock_4 = RepBlock(ERBlock_4, filters=channels_list[3], n=num_repeats[3], block=block, weight_decay=weight_decay)

    # 40, 40, 256 -> 20, 20, 512 ---------------- yolov6s
    ERBlock_5 = block(ERBlock_4, filters=channels_list[4], kernel_size=3, stride=2, weight_decay=weight_decay)
    ERBlock_5 = RepBlock(ERBlock_5, filters=channels_list[4], n=num_repeats[4], block=block, weight_decay=weight_decay)
    ERBlock_5 = SimSPPF(ERBlock_5, filters=channels_list[4], kernel_size=5, weight_decay=weight_decay)

    # [(80, 80, 128), (40, 40, 256), (20, 20, 512)]
    return ERBlock_3, ERBlock_4, ERBlock_5




