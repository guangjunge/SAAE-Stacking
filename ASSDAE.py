import os
from keras.layers import Input, GaussianNoise, Attention
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import utils
import keras
from keras.layers import Input, Dense
from tensorflow.keras.layers import Input, Dense
import numpy as np
import random
import tensorflow as tf
from keras import regularizers
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Softmax
import tensorflow_addons as tfa


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)


class ASSDAE:
    def __init__(self, input_dim, output_dim, hidden_dim, spar_arg):
        self.name = 'MalMem2022'
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._spar_arg = spar_arg
        self._encoder = self.build_encoder()
        self._decoder = self.build_decoder()
        self._autoencoder = self.build_autoencoder()

    def self_attention_layer(self, input):  # 注意力机制层
        # 通过对查询（query）和键（key）进行不同的处理，例如使用不同的全连接层，以确保它们具有不同的表示和语义。
        # 然后，通过计算它们之间的相似度得分和注意力权重，模型可以根据查询的特征选择性地关注键的不同部分。

        query = Dense(input.shape[1], activation=None)(input)

        key = Dense(input.shape[1], activation=None)(input)

        values = Dense(input.shape[1], activation=None)(input)

        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / (values.shape[1] ** 0.5)
        distribution = tf.nn.softmax(scores)

        return tf.matmul(distribution, values)

    def build_encoder(self):
        input_layer = Input(shape=(self._input_dim,), name='encoder_input_layer')
        encoder = GaussianNoise(0.01, name='addGaussianNoise')(input_layer)  # 添加高斯随机噪声

        # 堆叠多层隐藏层
        for dim in self._hidden_dim:
            encoder = Dense(dim, activation='selu', name='hidden_' + str(dim))(encoder)

        encoder = self.self_attention_layer(encoder) # query, key, value都为x
        encoder_output = Dense(self._output_dim, activation='selu', name='eccoder_output_layer')(encoder)
        return keras.Model(inputs=input_layer, outputs=encoder_output)

    def build_decoder(self):
        input_layer = Input(shape=(self._output_dim,), name='decoder_input_layer')
        decoder = input_layer
        for dim in reversed(self._hidden_dim):
            decoder = Dense(dim, activation='selu', name='hidden_' + str(dim))(decoder)

        decoder_output = Dense(self._input_dim, activation='softmax', name='decoder_output_layer')(decoder)
        return keras.Model(inputs=input_layer, outputs=decoder_output)

    def build_autoencoder(self):
        input_layer = Input(shape=(self._input_dim,), name='input_layer')
        encoder_output = self._encoder(input_layer)
        decoder_output = self._decoder(encoder_output)

        autoencoder = keras.Model(inputs=input_layer, outputs=decoder_output, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def get_encoder_output(self, x):
        encoder_output = self._encoder.predict(x)
        return encoder_output

    def get_decoder_output(self, x):
        decoder_output = self._decoder.predict(x)
        return decoder_output

    def summary(self):
        self._decoder.summary()
        self._encoder.summary()
        saveDir = './model/'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        utils.plot_model(self._encoder, saveDir + self._input_dim + '_encoder.png', show_shapes=True, show_dtype=True,
                         show_layer_names=True)
        utils.plot_model(self._decoder, saveDir + self._input_dim + '_decoder.png', show_shapes=True, show_dtype=True,
                         show_layer_names=True)

    def train(self, x_train, epochs=100, batch_size=32, validation_split=0.2, callback = None):
        if not callback:
            history = self._autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                                            validation_split=validation_split, shuffle=False)
        else:
            history = self._autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                                        validation_split=validation_split, shuffle=False, callbacks=[callback])
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        saveDir = './AAN_loss_plot/'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        plt.savefig(saveDir + self.name+'_loss_plot.png')  # 保存图像为PNG格式
        plt.show()
