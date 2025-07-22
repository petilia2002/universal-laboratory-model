import tensorflow as tf

from keras import layers
import tensorflow.python.keras.backend as K

MASKED_VALUE = -1

# Scale GPT-like embedding by the real value.
# No bias applied here.


class RealValueLayer(tf.keras.Layer):

    def __init__(self, embedding_size, **kwargs):
        super(RealValueLayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def build(self, input_shape):
        pass

    def call(self, inputs):

        gpt = inputs[0]
        v = inputs[1]

        v = tf.reshape(v, shape=(-1, v.shape[1], 1))
        v = tf.repeat(v, repeats=[self.embedding_size], axis=2)

        y = gpt * v
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# Mask for the encoder.


class MaskLayerLeft(layers.Layer):

    def __init__(self, **kwargs):
        super(MaskLayerLeft, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskLayerLeft, self).build(input_shape)

    def call(self, x):

        length = K.shape(x)[1]
        rank = tf.ones(shape=(1, length), dtype="float32")
        y = K.expand_dims(x, axis=-1)

        mask = K.dot(y, rank)
        return tf.transpose(mask, (0, 2, 1))


# Mask for the decoder.


class MaskLayerRight(layers.Layer):

    def __init__(self, **kwargs):
        super(MaskLayerRight, self).__init__(**kwargs)

    def call(self, x):

        right = x[0]
        left = x[1]

        length = K.shape(right)[1]
        rank = tf.ones(shape=(1, length), dtype="float32")
        y = K.expand_dims(left, axis=-1)

        mask = K.dot(y, rank)
        return tf.transpose(mask, (0, 2, 1))
