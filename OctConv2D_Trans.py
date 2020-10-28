from tensorflow.keras import layers
import tensorflow.keras.backend as K


# Transpose Octave Convolution Layer
class OctConv2D_Transpose(layers.Layer):
    """
    Define the Transpose OctConv2D which can replace the Transpose Conv2D layer.

    Paper:
    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution. (2019)
    """
    def __init__(self, filters, alpha, kernel_size=(2, 2), strides=(2, 2),
                 padding="valid", kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, kernel_constraint=None,
                 **kwargs):
        """
        :param filters: # output channels for low + high
        :param alpha: Low to high channels ratio (alpha=0 -> High channels only, alpha=1 -> Low channels only)
        :param kernel_size: 2x2 by default
        :param strides: 2x2 by default
        :param padding: "valid" by default
        :param kernel_initializer: "glorot_uniform" by default
        :param kernel_regularizer: "None" by default, you can specify one
        :param kernel_constraint: "None" by default, you can specify one
        :param kwargs: other keyword arguments
        """
        assert 0 <= alpha <= 1
        assert filters > 0 and isinstance(filters, int)
        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        # optional values
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        # Low Channels
        self.low_channels = int(self.filters * self.alpha)
        # High Channels
        self.high_channels = self.filters - self.low_channels

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        # assertion for high channel inputs
        assert input_shape[0][1] // 2 >= self.kernel_size[0]
        assert input_shape[0][2] // 2 >= self.kernel_size[1]
        # assertion for low channel inputs
        assert input_shape[0][1] // input_shape[1][1] == 2
        assert input_shape[0][2] // input_shape[1][2] == 2
        # "channels last" format for TensorFlow
        assert K.image_data_format() == "channels_last"
        # input channels
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        # High Channels -> High Channels
        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel",
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)
        # High Channels -> Low Channels
        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low Channels -> High Channels
        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low Channels -> Low Channels
        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)

        # output shape
        self.high_out_shape, self.low_out_shape = self.compute_output_shape(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        # Input = [X^H, X^L]
        assert len(inputs) == 2
        high_input, low_input = inputs
        # Transpose Convolution: High Channels -> High Channels
        high_to_high = K.conv2d_transpose(high_input, self.high_to_high_kernel, output_shape=self.high_out_shape,
                                          strides=self.strides, padding=self.padding,
                                          data_format="channels_last")
        # Transpose Convolution: High Channels -> Low Channels
        high_to_low = K.pool2d(high_input, (2, 2), strides=(2, 2), pool_mode="avg")
        high_to_low = K.conv2d_transpose(high_to_low, self.high_to_low_kernel, output_shape=self.low_out_shape,
                                         strides=self.strides, padding=self.padding,
                                         data_format="channels_last")
        # Transpose Convolution: Low Channels -> High Channels
        # Note: there is intermediate output size
        high_out_channels = self.high_out_shape[3]
        low_out_N, low_out_W, low_out_H = self.low_out_shape[:3]
        intermediate_shape = (low_out_N, low_out_W, low_out_H, high_out_channels)
        low_to_high = K.conv2d_transpose(low_input, self.low_to_high_kernel, output_shape=intermediate_shape,
                                         strides=self.strides, padding=self.padding,
                                         data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, 2, axis=1)  # Nearest Neighbor Upsampling
        low_to_high = K.repeat_elements(low_to_high, 2, axis=2)
        # Transpose Convolution: Low Channels -> Low Channels
        low_to_low = K.conv2d_transpose(low_input, self.low_to_low_kernel, output_shape=self.low_out_shape,
                                        strides=self.strides, padding=self.padding,
                                        data_format="channels_last")
        # Cross Add
        high_add = high_to_high + low_to_high
        low_add = high_to_low + low_to_low
        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes

        high_out_shape = (*high_in_shape[:3], self.high_channels)
        high_N, high_W, high_H, high_C = high_out_shape
        high_out_shape = (high_N, 2 * high_W, 2 * high_H, high_C)

        low_out_shape = (*low_in_shape[:3], self.low_channels)
        low_N, low_W, low_H, low_C = low_out_shape
        low_out_shape = (low_N, 2 * low_W, 2 * low_H, low_C)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        return out_config
