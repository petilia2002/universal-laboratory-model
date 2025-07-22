import numpy as np

import tensorflow as tf
import tensorflow.python.keras.backend as K

from keras import Model
from keras.layers import Input, Dense, Embedding
from keras.layers import Flatten, MultiHeadAttention, TimeDistributed
from keras.optimizers import Adam

from layers import RealValueLayer, MaskLayerLeft, MaskLayerRight, MASKED_VALUE

# Multitask loss, some outputs may be omitted.


def cbcLoss(y_true, y_pred):
    mask = K.cast_to_floatx(K.not_equal(y_true, MASKED_VALUE))
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    r = y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred)
    return -tf.reduce_mean(r * mask)


# Basic ULM Model.


def createUlm(embeddings_file, N_train, N_pred):

    gpt = np.load(embeddings_file)
    dims = gpt.shape
    emb_size = dims[1]

    # First create an embedding network, this model is not
    # supposed to be trainable.

    l_yandex = Input(name="input_analytes", shape=(None,))
    l_embed = Embedding(name="embedding", input_dim=dims[0], output_dim=dims[1])(
        l_yandex
    )

    embed = Model(l_yandex, l_embed, name="gpt")
    embed.set_weights([gpt])
    embed.trainable = False

    # Main network code

    l_in = Input(name="values", shape=(N_train,))
    l_words = Input(name="analytes", shape=(N_train,))
    l_mask = Input(name="mask", shape=(N_train,))

    l_pred = Input(name="outcomes", shape=(N_pred,))

    l_emb = embed(l_words)
    l_tests = embed(l_pred)
    embed.trainable = False

    l_m1 = MaskLayerLeft()(l_mask)
    l_m2 = MaskLayerRight()([l_pred, l_mask])

    l_v = RealValueLayer(embedding_size=emb_size, name="real_value_embedding")(
        [l_emb, l_in]
    )

    # Add trainable bias to the input values.
    l_pos = Embedding(name="yandex_bias", input_dim=dims[0], output_dim=emb_size)(
        l_words
    )
    l_v = l_v + l_pos

    # Add trainable bias to the ouytput values.
    l_p = Embedding(name="yandex_bias_pred", input_dim=dims[0], output_dim=emb_size)(
        l_pred
    )
    l_tests = l_tests + l_p

    l_enc = MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.1)(
        query=l_v, value=l_v, key=l_v, attention_mask=l_m1
    )
    l_dec = MultiHeadAttention(num_heads=8, key_dim=16, dropout=0.1)(
        query=l_tests, value=l_enc, key=l_enc, attention_mask=l_m2
    )

    l_o = TimeDistributed(Dense(units=16, activation="relu"))(l_dec)
    l_f = Flatten()(l_o)

    l_out_cbc = Dense(units=N_pred, activation="sigmoid")(l_f)

    model = Model([l_in, l_words, l_mask, l_pred], l_out_cbc)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=cbcLoss)
    model.summary()

    return model


# Vanilla shallow network, single task.


def createSingle(N_train):

    l_in = Input(name="values", shape=(N_train,))
    l_f = Dense(units=256, activation="relu")(l_in)
    l_out_cbc = Dense(units=1, activation="sigmoid")(l_f)

    model = Model(l_in, l_out_cbc)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy")
    model.summary()

    return model


# Multitask vanilla shallow network.


def createMultiTask(N_train, N_pred):

    l_in = Input(name="values", shape=(N_train,))
    l_f = Dense(units=256, activation="relu")(l_in)
    l_out_cbc = Dense(units=N_pred, activation="sigmoid")(l_f)

    model = Model(l_in, l_out_cbc)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=cbcLoss)
    model.summary()

    return model
