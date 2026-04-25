import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Layer
from tensorflow.keras import backend as K


class FeatureAttention(Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        self.W_q = self.add_weight(name="query", shape=(d, self.units), initializer="glorot_uniform", trainable=True)
        self.W_k = self.add_weight(name="key", shape=(d, self.units), initializer="glorot_uniform", trainable=True)
        self.W_v = self.add_weight(name="value", shape=(d, self.units), initializer="glorot_uniform", trainable=True)
        self.W_o = self.add_weight(name="output", shape=(self.units, d), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        Q = K.dot(x, self.W_q)
        K_m = K.dot(x, self.W_k)
        V = K.dot(x, self.W_v)
        scores = K.softmax(K.dot(Q, K.transpose(K_m)) / K.sqrt(K.cast(self.units, "float32")))
        return K.dot(K.dot(scores, V), self.W_o)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def create_model(input_dim=8):
    inp = Input((input_dim,))
    x = Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(inp)
    x = Dropout(0.4)(BatchNormalization()(x))
    attn = FeatureAttention(64)(x)
    x = Concatenate()([x, attn])
    x = Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(BatchNormalization()(x))
    x = Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(BatchNormalization()(x))
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(BatchNormalization()(x))
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(0.0005),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
