from nets.backbone import RepBlock, SimConv, RepVGGBlock, EfficientRep
from nets.loss import get_yolo_loss
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Conv2DTranspose, Concatenate, Input, Layer, Conv2D, BatchNormalization, Lambda
)

# 上采样
def Transpose(inputs, filters, kernel_size=2, stride=2):
    return Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride, use_bias=True)(inputs)

class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def Conv(inputs, filters, kernel_size, stride, groups=1, bias=False, padding='valid', weight_decay=5e-4):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, groups=groups, use_bias=bias,
               kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = SiLU()(x)
    return x


def RepPANNeck(inputs, channels_list=None, num_repeats=None, block=RepVGGBlock, weight_decay=5e-4):
    # [(80, 80, 128), (40, 40, 256), (20, 20, 512)]
    x2, x1, x0 = inputs
    fpn_out0 = SimConv(x0, filters=channels_list[5], kernel_size=1, stride=1, padding='valid', weight_decay=weight_decay)
    upsample_feat0 = Transpose(fpn_out0, channels_list[5])
    f_concat_layer0 = Concatenate(axis=-1)([upsample_feat0, x1])
    f_out0 = RepBlock(f_concat_layer0, filters=channels_list[5], n=num_repeats[5], block=block, weight_decay=weight_decay)

    fpn_out1 = SimConv(f_out0, filters=channels_list[6], kernel_size=1, stride=1, padding='valid', weight_decay=weight_decay)
    upsample_feat1 = Transpose(fpn_out1, channels_list[6])
    f_concat_layer1 = Concatenate(axis=-1)([upsample_feat1, x2])
    pan_out2 = RepBlock(f_concat_layer1, filters=channels_list[6], n=num_repeats[6], block=block, weight_decay=weight_decay)

    down_feat1 = SimConv(pan_out2, filters=channels_list[7], kernel_size=3, stride=2, padding='same', weight_decay=weight_decay)
    p_concat_layer1 = Concatenate(axis=-1)([down_feat1, fpn_out1])
    pan_out1 = RepBlock(p_concat_layer1, filters=channels_list[8], n=num_repeats[7], block=block, weight_decay=weight_decay)

    down_feat0 = SimConv(pan_out1, filters=channels_list[9], kernel_size=3, stride=2, padding='same', weight_decay=weight_decay)
    p_concat_layer2 = Concatenate(axis=-1)([down_feat0, fpn_out0])
    pan_out0 = RepBlock(p_concat_layer2, filters=channels_list[10], n=num_repeats[8], block=block, weight_decay=weight_decay)
    
    outputs = [pan_out2, pan_out1, pan_out0] # [(80, 80, 64), (40, 40, 128), (20, 20, 256)]
    return outputs



def Detect(inputs, num_classes=80, channels_list=None, num_anchors=None, weight_decay=5e-4):
    x2, x1, x0 = inputs # [(80, 80, 64), (40, 40, 128), (20, 20, 256)]
    
    x2 = Conv(x2, filters=channels_list[6], kernel_size=1, stride=1)
    cls_x2 = x2
    reg_x2 = x2
    cls_feat_x2 = Conv(cls_x2, channels_list[6], kernel_size=3, stride=1, padding='same', weight_decay=weight_decay)
    cls_output_x2 = Conv2D(filters=num_classes * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(cls_feat_x2)
    reg_feat_x2 = Conv(reg_x2, filters=channels_list[6], kernel_size=3, stride=1, padding='same', weight_decay=weight_decay)
    reg_output_x2 = Conv2D(filters=4 * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(reg_feat_x2)
    obj_output_x2 = Conv2D(filters=1 * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(reg_feat_x2)
    x2 = Concatenate(axis=-1)([reg_output_x2, obj_output_x2, cls_output_x2])
    
    
    x1 = Conv(x1, filters=channels_list[8], kernel_size=1, stride=1, weight_decay=weight_decay)
    cls_x1 = x1
    reg_x1 = x1
    cls_feat_x1 = Conv(cls_x1, channels_list[8], kernel_size=3, stride=1, padding='same', weight_decay=weight_decay)
    cls_output_x1 = Conv2D(filters=num_classes * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(cls_feat_x1)
    reg_feat_x1 = Conv(reg_x1, filters=channels_list[8], kernel_size=3, stride=1, padding='same', weight_decay=weight_decay)
    reg_output_x1 = Conv2D(filters=4 * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(reg_feat_x1)
    obj_output_x1 = Conv2D(filters=1 * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(reg_feat_x1)
    x1 = Concatenate(axis=-1)([reg_output_x1, obj_output_x1, cls_output_x1])
    
    x0 = Conv(x0, filters=channels_list[10], kernel_size=1, stride=1, weight_decay=weight_decay)
    cls_x0 = x0
    reg_x0 = x0
    cls_feat_x0 = Conv(cls_x0, channels_list[10], kernel_size=3, stride=1, padding='same', weight_decay=weight_decay)
    cls_output_x0 = Conv2D(filters=num_classes * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(cls_feat_x0)
    reg_feat_x0 = Conv(reg_x0, filters=channels_list[10], kernel_size=3, stride=1, padding='same', weight_decay=weight_decay)
    reg_output_x0 = Conv2D(filters=4 * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(reg_feat_x0)
    obj_output_x0 = Conv2D(filters=1 * num_anchors, kernel_size=1, kernel_initializer=RandomNormal(stddev=0.02),kernel_regularizer=l2(weight_decay))(reg_feat_x0)
    x0 = Concatenate(axis=-1)([reg_output_x0, obj_output_x0, cls_output_x0])
    
    return [x2, x1, x0]



def Yolov6(inputs, num_classes=80, phi='s', weight_decay=5e-4):
    inputs = Input(shape=inputs)
    if phi == 's':
        depth_mul, width_mul = 0.33, 0.50
        num_repeat_backbone = [1, 6, 12, 18, 6]
        channels_list_backbone = [64, 128, 256, 512, 1024]
        num_repeat_neck = [12, 12, 12, 12]
        channels_list_neck = [256, 128, 128, 256, 256, 512]
        num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
        channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
        backbone = EfficientRep(inputs=inputs, channels_list=channels_list, num_repeats=num_repeat, weight_decay=weight_decay)
        neck = RepPANNeck(backbone, channels_list=channels_list, num_repeats=num_repeat, weight_decay=weight_decay)
        detect = Detect(neck, num_classes=num_classes, channels_list=channels_list, num_anchors=1, weight_decay=weight_decay)
        return Model(inputs, detect)
    elif phi == 'tiny':
        depth_mul, width_mul = 0.25, 0.50
        num_repeat_backbone = [1, 6, 12, 18, 6]
        channels_list_backbone = [64, 128, 256, 512, 1024]
        num_repeat_neck = [12, 12, 12, 12]
        channels_list_neck = [256, 128, 128, 256, 256, 512]
        num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
        channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
        backbone = EfficientRep(inputs=inputs, channels_list=channels_list, num_repeats=num_repeat, weight_decay=weight_decay)
        neck = RepPANNeck(backbone, channels_list=channels_list, num_repeats=num_repeat, weight_decay=weight_decay)
        detect = Detect(neck, num_classes=num_classes, channels_list=channels_list, num_anchors=1, weight_decay=weight_decay)
        return Model(inputs, detect)
    elif phi == 'n':
        depth_mul, width_mul = 0.33, 0.25
        num_repeat_backbone = [1, 6, 12, 18, 6]
        channels_list_backbone = [64, 128, 256, 512, 1024]
        num_repeat_neck = [12, 12, 12, 12]
        channels_list_neck = [256, 128, 128, 256, 256, 512]
        num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
        channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
        backbone = EfficientRep(inputs=inputs, channels_list=channels_list, num_repeats=num_repeat, weight_decay=weight_decay)
        neck = RepPANNeck(backbone, channels_list=channels_list, num_repeats=num_repeat, weight_decay=weight_decay)
        detect = Detect(neck, num_classes=num_classes, channels_list=channels_list, num_anchors=1, weight_decay=weight_decay)
        return Model(inputs, detect)


def make_divisible(x, divisor):
    import math
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

def get_train_model(model_body, input_shape, num_classes, iou_type='siou'):
    y_true = [Input(shape = (None, 5))]
    model_loss  = Lambda(
        get_yolo_loss(input_shape, len(model_body.output), num_classes, iou_type=iou_type), 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
    )([*model_body.output, *y_true])
    
    model       = Model([model_body.input, *y_true], model_loss)
    return model