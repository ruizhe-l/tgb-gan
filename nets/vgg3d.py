import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers

class VGG3D(Model):
    def __init__(self, n_layer, root_filters, kernal_size=3, pool_size=2, use_bn=True, use_res=True, padding='SAME'):
        super().__init__()
        self.dw_layers = dict()
        self.max_pools = dict()
        for layer in range(n_layer):
            filters = 2**layer*root_filters
            dict_key = str(n_layer-layer-1)
            dw = _DownSampling(filters, kernal_size, 'dw_%d'%layer, use_bn, use_res)
            self.dw_layers[dict_key] = dw
            pool = layers.MaxPool3D(pool_size, padding=padding)
            self.max_pools[dict_key] = pool

        self.flat = layers.Flatten()
        self.f1 = layers.Dense(2048, use_bias=True, name='f1')
        self.f2 = layers.Dense(512, use_bias=True, name='f2')
        self.f3 = layers.Dense(64, use_bias=True, name='f3')
        self.f_out = layers.Dense(1, use_bias=False, name='f_out')
        

    @tf.function
    def __call__(self, x_in, drop_rate, training):
        dw_tensors = dict()
        x = x_in
        n_layer = len(self.dw_layers)
        for i in range(n_layer):
            dict_key = str(n_layer-i-1)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, drop_rate, training)
            x = dw_tensors[dict_key]
            x = self.max_pools[dict_key](x)

        x = self.flat(x)
        x = self.f1(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.f2(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.f3(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, drop_rate)
        x = self.f_out(x)
        # x = tf.nn.relu(x)
        return x



class _DownSampling(layers.Layer):

    def __init__(self, filters, kernel_size, name, use_bn=True, use_res=True, padding='SAME', use_bias=True):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        # stddev = np.sqrt(2/(kernel_size**2*filters))
        self.conv1 = layers.Conv3D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                    # kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                    name='conv1')
        self.conv2 = layers.Conv3D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                    # kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                    name='conv2')
        
        if use_bn:
            self.bn1 = layers.BatchNormalization(momentum=0.99, name='bn1')
            self.bn2 = layers.BatchNormalization(momentum=0.99, name='bn2')
        if use_res:
            self.res = _Residual('res')

    def __call__(self, x_in, drop_rate, training=True):
        # conv1
        x = self.conv1(x_in)
        # x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        # conv2
        x = self.conv2(x)
        # x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn2(x, training=training)
        
        if self.use_res:
            x = self.res(x_in, x)
        x = tf.nn.relu(x)

        return x


class _Residual(layers.Layer):
    def __init__(self, name_scope):
        super(_Residual, self).__init__(name=name_scope)
    
    def call(self, x1, x2):
        if x1.shape[-1] < x2.shape[-1]:
            x = tf.concat([x1, tf.zeros(list(x1.shape[:-1]) + [x2.shape[-1] - x1.shape[-1]])], axis=-1)
        else:
            x = x1[..., :x2.shape[-1]]
        x = x + x2
        return x
