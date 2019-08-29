from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model


class UNetGenerator:
    def __init__(self, img_shape, filters, channels, output_activation, skip_connections):
        self.img_shape = img_shape
        self.filters = filters
        self.channels = channels
        self.output_activation = output_activation
        self.skip_connection = skip_connections

    def build(self):
        def conv2d(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size,
                       strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = Conv2DTranspose(filters, kernel_size=f_size, strides=(2, 2),
                                padding='same', activation='linear')(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1,
                       padding='same', activation='relu')(u)

            u = BatchNormalization(momentum=0.8)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            if self.skip_connection:
                u = Concatenate()([u, skip_input])

            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling: 7 x stride of 2 --> x1/128 downsampling
        d1 = conv2d(d0, self.filters, bn=False)
        d2 = conv2d(d1, self.filters * 2)
        d3 = conv2d(d2, self.filters * 4)
        d4 = conv2d(d3, self.filters * 8)
        d5 = conv2d(d4, self.filters * 8)
        d6 = conv2d(d5, self.filters * 8)
        d7 = conv2d(d6, self.filters * 8)

        # Upsampling: 6 x stride of 2 --> x64 upsampling
        u1 = deconv2d(d7, d6, self.filters * 8)
        u2 = deconv2d(u1, d5, self.filters * 8)
        u3 = deconv2d(u2, d4, self.filters * 8)
        u4 = deconv2d(u3, d3, self.filters * 4)
        u5 = deconv2d(u4, d2, self.filters * 2)
        u6 = deconv2d(u5, d1, self.filters)
        u7 = Conv2DTranspose(self.channels, kernel_size=4, strides=(2, 2),
                             padding='same', activation='linear')(u6)

        # added conv layers after the deconvs to avoid the pixelated outputs
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding='same',
                            activation=self.output_activation)(u7)

        return Model(d0, output_img)


class Discriminator:
    def __init__(self, img_shape, filters, num_layers, conditional=False):
        self.img_shape = img_shape
        self.filters = filters
        self.num_layers = num_layers
        self.conditional = conditional

    def build(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        if self.conditional:
            input_inputs = Input(shape=self.img_shape)
            input_targets = Input(shape=self.img_shape)
            discriminator_input_image = Concatenate(axis=-1)([input_targets, input_inputs])
            discriminator_input_list = [input_targets, input_inputs]
        else:
            input_inputs = Input(shape=self.img_shape)
            discriminator_input_image = input_inputs
            discriminator_input_list = [input_inputs]

        # 4 d_layers with stride of 2 --> output is 1/16 in each dimension
        d = d_layer(discriminator_input_image, self.filters, bn=False)

        for i in range(self.num_layers - 1):
            d = d_layer(d, self.filters * (2 ** (i + 1)))

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d)

        return Model(discriminator_input_list, validity)
