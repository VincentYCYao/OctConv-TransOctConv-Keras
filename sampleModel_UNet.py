from tensorflow.keras import layers
from tensorflow.keras import models

import OctConv2D
import OctConv2D_Trans


# Define a OctConv-BatchNormalization-ReLU block
def OctConv2D_BN_ReLU(input_tensor, num_filters, alpha, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    [high, low] = OctConv2D(num_filters, alpha, kernel_size, strides, padding)(input_tensor)
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)
    return [high, low]


# Define a pre-activation block
def BN_ReLU_OctConv2D(input_tensor, num_filters, alpha, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    [high, low] = input_tensor
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)
    [high, low] = OctConv2D(num_filters, alpha, kernel_size, strides, padding)([high, low])
    return [high, low]


# Define a BatchNormalization-ReLU block
def BN_ReLU(input_tensor):
    [high, low] = input_tensor
    high = layers.BatchNormalization()(high)
    high = layers.Activation("relu")(high)
    low = layers.BatchNormalization()(low)
    low = layers.Activation("relu")(low)
    return [high, low]


# Add the outputs of OctConv
def add_octconv(in1, in2):
    high1, low1 = in1
    high2, low2 = in2
    high = layers.Add()([high1, high2])
    low = layers.Add()([low1, low2])
    return [high, low]


# Average-pooling to the outputs from OctConv
def average_pooling_octconv(input_tensor):
    high, low = input_tensor
    high_out = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(high)
    low_out = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(low)
    return [high_out, low_out]


# Concatenation of outputs from OctConv
def concatenate_octconv(in1, in2):
    high1, low1 = in1
    high2, low2 = in2
    high = layers.concatenate([high1, high2], axis=-1)
    low = layers.concatenate([low1, low2], axis=-1)
    return [high, low]


# Define OctConv-equipped residual block
def octconv_residual_block(input_tensor, num_filters, alpha, n_repeat, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    """
    FEATURES:
    - Define convolution block with Pre-activation shortcut (paper: Identity mappings in deep residual networks)
    - Combine short and long residual shortcut
    """
    encoder_1 = OctConv2D(num_filters, alpha, kernel_size, strides, padding)(input_tensor)
    encoder_x = encoder_1
    # TODO: modify the for-loop to repeat the following block for n times (n=5 here)
    for i in range(n_repeat):
        encoder_tmp1 = BN_ReLU_OctConv2D(encoder_x, num_filters, alpha, kernel_size, strides, padding)
        encoder_tmp2 = BN_ReLU_OctConv2D(encoder_tmp1, num_filters, alpha, kernel_size, strides, padding)
        # short residual shortcut: encoder_x -> encoder_tmp2
        encoder_skip1 = add_octconv(encoder_x, encoder_tmp2)
        encoder_tmp3 = BN_ReLU_OctConv2D(encoder_skip1, num_filters, alpha, kernel_size, strides, padding)
        encoder_tmp4 = BN_ReLU_OctConv2D(encoder_tmp3, num_filters, alpha, kernel_size, strides, padding)
        # short residual shortcut: encoder_skip1 -> encoder_tmp4
        # long residual shortcut: encoder_x -> encoder_tmp4
        encoder_x = add_octconv(encoder_x, add_octconv(encoder_skip1, encoder_tmp4))
    encoder_end = BN_ReLU(encoder_x)
    return encoder_end


# Define encoder block with residual shortcut and OctConv
def octconv_encoder_block(input_tensor, num_filters, alpha, n_repeat, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    encoder = octconv_residual_block(input_tensor, num_filters, alpha, n_repeat, kernel_size, strides, padding)
    encoder_pool = average_pooling_octconv(encoder)
    return encoder_pool, encoder


# Define decoder block
def residual_decoder_block(input_tensor, concat_tensor, num_filters, alpha, n_repeat, kernel_size=(3, 3),
                           strides=(1, 1), octconv_padding="same", octconvT_padding='valid'):
    decoder = OctConv2D_Trans(num_filters, alpha, kernel_size=(2, 2), strides=(2, 2), padding=octconvT_padding)(
        input_tensor)
    # concatenated with features from encoder block
    decoder = concatenate_octconv(decoder, concat_tensor)
    decoder_x = OctConv2D(num_filters, alpha, kernel_size, strides, octconv_padding)(decoder)

    # TODO: modify the for-loop to repeat the following block for n times (n=5 here)
    for i in range(n_repeat):
        decoder_tmp1 = BN_ReLU_OctConv2D(decoder_x, num_filters, alpha, kernel_size, strides, octconv_padding)
        decoder_tmp2 = BN_ReLU_OctConv2D(decoder_tmp1, num_filters, alpha, kernel_size, strides, octconv_padding)
        # short residual shortcut: decoder_x -> decoder_tmp2
        decoder_skip1 = add_octconv(decoder_x, decoder_tmp2)
        decoder_tmp3 = BN_ReLU_OctConv2D(decoder_skip1, num_filters, alpha, kernel_size, strides, octconv_padding)
        decoder_tmp4 = BN_ReLU_OctConv2D(decoder_tmp3, num_filters, alpha, kernel_size, strides, octconv_padding)
        # short residual shortcut: decoder_skip1 -> decoder_tmp4
        # long residual shortcut: decoder_x -> decoder_tmp4
        decoder_x = add_octconv(decoder_x, add_octconv(decoder_skip1, decoder_tmp4))
    decoder_end = BN_ReLU(decoder_x)
    return decoder_end


def UNet_octconv(alpha, in_shape):
    """
    Construct U-Net with OctConv layers, pre-activation shortcut, and average-pooling

    ARGUMENTS:
    :param alpha: Low to high channels ratio (alpha=0 -> High channels only, alpha=1 -> Low channels only)
    :param in_shape: input shape

    OUTPUT:
    :return: the U-Net model
    """
    # construct input with high and low features (assume the input shape is 512x512)
    input_high = layers.Input(shape=in_shape, name='inputs')  # 512
    input_low = layers.AveragePooling2D(2)(input_high)  # 256
    input_1 = [input_high, input_low]  # 512, 256
    # the encoding pathway
    encoder0_pool, encoder0 = octconv_encoder_block(input_1, 16, alpha, 5)  # (256, 128) (512, 256)
    encoder1_pool, encoder1 = octconv_encoder_block(encoder0_pool, 32, alpha, 5)  # (128, 64) (256, 128)
    encoder2_pool, encoder2 = octconv_encoder_block(encoder1_pool, 64, alpha, 5)  # (64, 32) (128, 64)
    encoder3_pool, encoder3 = octconv_encoder_block(encoder2_pool, 128, alpha, 5)  # (32, 16) (64, 32)
    encoder4_pool, encoder4 = octconv_encoder_block(encoder3_pool, 256, alpha, 5)  # (16, 8) (32, 16)
    # center
    center = octconv_residual_block(encoder4_pool, 512, alpha, 5)  # 16, 8
    # the decoding pathway
    decoder4 = residual_decoder_block(center, encoder4, 256, alpha, 5)  # 32, 16
    decoder3 = residual_decoder_block(decoder4, encoder3, 128, alpha, 5)  # 64, 32
    decoder2 = residual_decoder_block(decoder3, encoder2, 64, alpha, 5)  # 128, 64
    decoder1 = residual_decoder_block(decoder2, encoder1, 32, alpha, 5)  # 256, 128
    decoder0 = residual_decoder_block(decoder1, encoder0, 16, alpha, 5)  # 512, 256
    # add OctConv-BN-ReLU
    segment1 = OctConv2D_BN_ReLU(decoder0, 16, alpha)  # 512, 256
    # add Conv2D layer to the high and low channel outputs
    seg_high, seg_low = segment1
    out_high = layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same")(seg_high)
    out_low = layers.Conv2D(filters=2, kernel_size=(3, 3), padding="same")(seg_low)
    # up-sample the low channel output
    out_low2high = layers.UpSampling2D((2, 2), "channels_last", "bilinear")(out_low)
    # add the up-sampled low channel output to high channel output
    output_high = layers.Add()([out_high, out_low2high])
    output_high = layers.BatchNormalization()(output_high)
    output_high = layers.Activation("relu")(output_high)  # 512
    # add final Conv2D layer with 'sigmoid' activation
    output_high = layers.Conv2D(1, (1, 1), activation='sigmoid')(output_high)
    # get model
    model = models.Model(inputs=input_high, outputs=output_high)
    return model
