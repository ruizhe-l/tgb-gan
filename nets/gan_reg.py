import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
from model.vgg3d import VGG3D


class GEN_REG(Model):
    def __init__(self, gen, reg):
        super().__init__()
        self.gen = gen
        self.reg = reg

    def __call__(self, x_in, drop_rate, training=False):
        x1 = self.gen(x_in, drop_rate, training)
        x2 = self.reg(x1, drop_rate, training)
        return x1, x2


class Discriminator(Model):
    def __init__(self, kernal_size=3, use_bn=True, padding='SAME'):
        super().__init__()

        self.dw1 = _DownSampling(64, kernal_size, 'dw1', use_bn=use_bn, padding=padding)
        self.dw2 = _DownSampling(128, kernal_size, 'dw2', use_bn=use_bn, padding=padding)
        self.dw3 = _DownSampling(256, kernal_size, 'dw3', use_bn=use_bn, padding=padding)
        self.dw4 = _DownSampling(512, kernal_size, 'dw4', use_bn=use_bn, padding=padding)

        self.conv_out = layers.Conv3D(1, kernal_size, strides=1, padding=padding, use_bias=False, name='conv_out')

    def __call__(self, x_in, drop_rate=0, training=False):
        x = self.dw1(x_in, drop_rate, training)
        x = self.dw2(x, drop_rate, training)
        x = self.dw3(x, drop_rate, training)
        x = self.dw4(x, drop_rate, training)

        x = self.conv_out(x)
        # x = tf.nn.sigmoid(x)

        return x

class Gen_encoder(Model):
    def __init__(self, kernal_size=3, use_bn=True, padding='SAME'):
        super().__init__()


        self.dw1 = _DownSampling(64, kernal_size, 'dw1', use_bn=use_bn, padding=padding) # 48
        self.dw2 = _DownSampling(128, kernal_size, 'dw2', use_bn=use_bn, padding=padding) # 24
        self.dw3 = _DownSampling(256, kernal_size, 'dw3', use_bn=use_bn, padding=padding) # 12
        self.dw4 = _DownSampling(512, kernal_size, 'dw4', use_bn=use_bn, padding=padding) # 6
        self.dw5 = _DownSampling(512, kernal_size, 'dw5', use_bn=use_bn, padding=padding) # 3
        
    @tf.function
    def __call__(self, x_in, drop_rate=0, training=False):
        x = self.dw1(x_in, drop_rate, training)
        x = self.dw2(x, drop_rate, training)
        x = self.dw3(x, drop_rate, training)
        x = self.dw4(x, drop_rate, training)
        x = self.dw5(x, drop_rate, training)

        return x

class Gen_decoder(Model):
    def __init__(self, output_channels, kernal_size=3, use_bn=True, padding='SAME'):
        super().__init__()

        self.up1 = _UpSampling(512, kernal_size, 'up1', use_bn=use_bn, padding=padding)
        self.up2 = _UpSampling(512, kernal_size, 'up2', use_bn=use_bn, padding=padding)
        self.up3 = _UpSampling(256, kernal_size, 'up3', use_bn=use_bn, padding=padding)
        self.up4 = _UpSampling(128, kernal_size, 'up4', use_bn=use_bn, padding=padding)
        self.up5 = _UpSampling(64, kernal_size, 'up5', use_bn=use_bn, padding=padding)

        self.conv_out = layers.Conv3D(output_channels, 1, padding=padding, use_bias=False, name='conv_out')

    @tf.function
    def __call__(self, x_in, drop_rate=0, training=False):

        x = self.up1(x_in, drop_rate, training)
        x = self.up2(x, drop_rate, training)
        x = self.up3(x, drop_rate, training)
        x = self.up4(x, drop_rate, training)
        x = self.up5(x, drop_rate, training)

        x = self.conv_out(x)
        x = tf.nn.relu(x)
        return x

class Generator(Model):

    def __init__(self, output_channels, kernal_size=3, use_bn=True, padding='SAME'):
        super().__init__()


        self.dw1 = _DownSampling(64, kernal_size, 'dw1', use_bn=use_bn, padding=padding) # 48
        self.dw2 = _DownSampling(128, kernal_size, 'dw2', use_bn=use_bn, padding=padding) # 24
        self.dw3 = _DownSampling(256, kernal_size, 'dw3', use_bn=use_bn, padding=padding) # 12
        self.dw4 = _DownSampling(512, kernal_size, 'dw4', use_bn=use_bn, padding=padding) # 6
        self.dw5 = _DownSampling(512, kernal_size, 'dw5', use_bn=use_bn, padding=padding) # 3
        # self.dw6 = _DownSampling(512, kernal_size, 'dw6', use_bn=use_bn, padding=padding)

        self.up1 = _UpSampling(512, kernal_size, 'up1', use_bn=use_bn, padding=padding)
        self.up2 = _UpSampling(512, kernal_size, 'up2', use_bn=use_bn, padding=padding)
        self.up3 = _UpSampling(256, kernal_size, 'up3', use_bn=use_bn, padding=padding)
        self.up4 = _UpSampling(128, kernal_size, 'up4', use_bn=use_bn, padding=padding)
        self.up5 = _UpSampling(64, kernal_size, 'up5', use_bn=use_bn, padding=padding)
        # self.up6 = _UpSampling(64, kernal_size, 'up6', use_bn=use_bn, padding=padding)
        self.conv_out = layers.Conv3D(output_channels, 1, padding=padding, use_bias=False, name='conv_out')

    @tf.function
    def __call__(self, x_in, drop_rate=0, training=False):
        x = self.dw1(x_in, drop_rate, training)
        x = self.dw2(x, drop_rate, training)
        x = self.dw3(x, drop_rate, training)
        x = self.dw4(x, drop_rate, training)
        x = self.dw5(x, drop_rate, training)
        # x = self.dw6(x, drop_rate, training)

        x = self.up1(x, drop_rate, training)
        x = self.up2(x, drop_rate, training)
        x = self.up3(x, drop_rate, training)
        x = self.up4(x, drop_rate, training)
        x = self.up5(x, drop_rate, training)
        # x = self.up6(x, drop_rate, training)

        x = self.conv_out(x)
        x = tf.nn.relu(x)
        return x

class _DownSampling(layers.Layer):

    def __init__(self, filters, kernel_size, name, pool_size=2, use_bn=True, use_res=True, padding='SAME', use_bias=True):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        stddev = np.sqrt(2/(kernel_size**2*filters))
        self.conv = layers.Conv3D(filters, kernel_size, strides=pool_size, padding=padding, use_bias=use_bias,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                    name='conv')
        
        if use_bn:
            self.bn = layers.BatchNormalization(momentum=0.99, name='bn')

    def __call__(self, x_in, drop_rate, training):
        # conv1
        x = self.conv(x_in)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x


class _UpSampling(layers.Layer):

    def __init__(self, filters, kernel_size, name, concat_or_add='concat', use_bn=True, padding='SAME', use_bias=True):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.concat_or_add = concat_or_add
        stddev = np.sqrt(2/(kernel_size**2*filters))
        self.deconv = layers.Conv3DTranspose(filters//2, kernel_size, strides=2, padding=padding, use_bias=use_bias,
                                            kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                            name='deconv')
        
        if use_bn:
            self.bn_deconv = layers.BatchNormalization(momentum=0.99, name='bn_deconv')

    def __call__(self, x_in, drop_rate, training):
        # deconv
        x = self.deconv(x_in)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn_deconv(x, training=training)
        x = tf.nn.relu(x)

        return x
