import os
from keras.layers import Input, GaussianNoise, MultiHeadAttention
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import utils
import keras
from keras.layers import Input, Dense
from tensorflow.keras.layers import Input, Dense
import numpy as np
import random
import tensorflow as tf

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

    def attention_layer(self, query, key, value):  # 注意力机制层
        # 通过对查询（query）和键（key）进行不同的处理，例如使用不同的全连接层，以确保它们具有不同的表示和语义。
        # 然后，通过计算它们之间的相似度得分和注意力权重，模型可以根据查询的特征选择性地关注键的不同部分。

        query = Dense(query.shape[1])(query)

        key = Dense(key.shape[1])(key)

        value = Dense(key.shape[1])(value)

        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / (value.shape[1] ** 0.5)
        distribution = tf.nn.softmax(scores)
        self._distribution = distribution  # 添加此行代码
        return tf.matmul(distribution, value)

    # def attention_layer(self, input):  # 注意力机制层
    #     # 通过对查询（query）和键（key）进行不同的处理，例如使用不同的全连接层，以确保它们具有不同的表示和语义。
    #     # 然后，通过计算它们之间的相似度得分和注意力权重，模型可以根据查询的特征选择性地关注键的不同部分。
    #     # 构建自注意力层
    #     # output即为经过全连接层处理后的输出
    #     query = Dense(input.shape[1])(input)
    #     key = Dense(input.shape[1])(input)
    #     value= Dense(input.shape[1])(input)
    #
    #     scores = tf.matmul(query, key)
    #     distribution = tf.nn.softmax(scores)
    #     return tf.matmul(value, distribution, transpose_b=True)

    def build_encoder(self):
        input_layer = Input(shape=(self._input_dim,), name='encoder_input_layer')
        encoder = GaussianNoise(10, name='addGaussianNoise')(input_layer)  # 添加高斯随机噪声
        encoder = self.attention_layer(encoder,encoder,encoder)  # 计算注意力权重
        # encoder = input_layer
        # 堆叠多层隐藏层
        for dim in self._hidden_dim:
            # encoder = self.attention_layer(encoder,encoder,encoder)  # 计算注意力权重
            encoder = Dense(dim, activation='selu', name='hidden_' + str(dim))(encoder)

        encoder_output = keras.layers.ActivityRegularization(l1=self._spar_arg)(encoder)  # 添加稀疏性
        # encoder_output = self.attention_layer(encoder_output, encoder_output,encoder_output)  # 计算注意力权重
        encoder_output = Dense(self._output_dim, activation='selu', name='eccoder_output_layer')(encoder_output)
        return keras.Model(inputs=input_layer, outputs=encoder_output)

    def build_decoder(self):
        input_layer = Input(shape=(self._output_dim,), name='decoder_input_layer')
        decoder = input_layer

        for dim in reversed(self._hidden_dim):
            decoder = Dense(dim, activation='selu', name='hidden_' + str(dim))(decoder)

        decoder = keras.layers.ActivityRegularization(l1=self._spar_arg)(decoder)  # 添加稀疏性
        decoder_output = Dense(self._input_dim, activation='selu', name='decoder_output_layer')(decoder)
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
        utils.plot_model(self._encoder, saveDir + '_encoder.pdf', show_shapes=True, show_dtype=True,
                         show_layer_names=True)
        utils.plot_model(self._decoder, saveDir + '_decoder.pdf', show_shapes=True, show_dtype=True,
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
        plt.savefig(saveDir + self.name+'_loss_plot.svg',format='svg', dpi=1200)  # 保存图像为PNG格式
        plt.show()
