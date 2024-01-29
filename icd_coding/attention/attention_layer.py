import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class Attention(Layer):

    def __init__(self, nb_classes, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.nb_classes = nb_classes

    def build(self, input_shape):
        self.feature_dims = input_shape[2]

        self.Wa = self.add_weight(shape=(self.nb_classes, self.feature_dims),
                                  initializer=initializers.get('glorot_uniform'),
                                  trainable=True,
                                  name='weights')
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        # inputs -> (batch_size, sentence_length, the number of feature dims)
        # inputs_trans -> (batch_size, the number of feature dims, sentence_length)
        inputs_trans = tf.transpose(inputs, [0, 2, 1])
        at = tf.matmul(self.Wa, inputs_trans)  # shape=[batch_size, the number of classes, sentence_length]

        # masking before softmax
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.tile(mask, [1, self.nb_classes, 1])
            mask = tf.cast(mask, K.floatx())
            at = at - (1 - mask) * 1e12

        attention_weights = K.softmax(at, axis=-1)

        # weighted sum
        # v = (batch_size, the number of classes, the number of feature dims)
        attention_adjusted_output = K.batch_dot(attention_weights, inputs)

        return attention_weights, attention_adjusted_output

    def get_config(self):
        config = super(Attention, self).get_config()
        config['nb_classes'] = self.nb_classes
        return config


class Label_Attention(Layer):

    def __init__(self, nb_classes, da, **kwargs):
        super(Label_Attention, self).__init__(**kwargs)
        self.nb_classes = nb_classes
        self.da = da

    def build(self, input_shape):
        self.feauture_dims = input_shape[-1]
        self.sentence_length = input_shape[1]

        # self.Wa = (the number of classes, the number of feature dims)
        self.W1 = self.add_weight(shape=(self.da, self.feauture_dims),
                                  initializer=initializers.get('glorot_uniform'),
                                  trainable=True,
                                  name='weights_1')
        self.W2 = self.add_weight(shape=(self.nb_classes, self.da),
                                  initializer=initializers.get('glorot_uniform'),
                                  trainable=True,
                                  name='weights_2')
        self.bias = self.add_weight(shape=(self.nb_classes, self.sentence_length),
                                    initializer="zeros",
                                    trainable=True,
                                    name='biases')

        super(Label_Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):

        return None

    def call(self, inputs, mask=None):
        # inputs = (batch_size, sentence_length, the number of LSTM dims)
        # inputs_trans = (batch_size, the number of LSTM dims, sentence_length)
        inputs_trans = tf.transpose(inputs, [0, 2, 1])

        # at = (batch_size, da, sent_len)
        at = tf.matmul(self.W1, inputs_trans)
        # at = (batch_size, class_num, sent_len)
        at = tf.matmul(self.W2, at)

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.tile(mask, [1, self.nb_classes, 1])
            mask = tf.cast(mask, K.floatx())
            at = at - (1 - mask) * 1e12

        # Softmax
        attention_weights = K.softmax(at, axis=-1)

        # weighted sum
        # v = (batch_size, the number of classes, the number of feature dims)
        attention_adjusted_output = K.batch_dot(attention_weights, inputs)

        return attention_weights, attention_adjusted_output

    def get_config(self):
        config = super(Label_Attention, self).get_config()
        config['nb_classes'] = self.nb_classes
        config['da'] = self.da
        return config

'''
class LSTM_Attention(tf.keras.layers.Layer):

    def __init__(self, class_num, **kwargs):
        super(LSTM_Attention, self).__init__(**kwargs)
        self.class_num = class_num

    def build(self, input_shape):
        self.num_dim_perword = input_shape[-1]
        self.sentence_length = input_shape[1]

        # self.Wa = (the number of classes, the number of LSTM dims)
        self.Wa = self.add_weight(shape=(self.class_num, self.num_dim_perword),
                                  initializer=initializers.get('glorot_uniform'),
                                  trainable=True,
                                  name='weights')
        self.bias = self.add_weight(shape=(self.class_num, self.sentence_length),
                                    initializer="zeros",
                                    trainable=True,
                                    name='biases')

        super(LSTM_Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):

        return None

    def call(self, inputs, mask=None):
        # inputs = (batch_size, sentence_length, the number of LSTM dims)
        # inputs_trans = (batch_size, the number of LSTM dims, sentence_length)
        inputs_trans = tf.transpose(inputs, [0, 2, 1])

        # at = (batch_size, the number of classes, sentence_length)
        # at = K.tanh(tf.matmul(self.Wa, inputs_trans) + self.bias)
        # at = tf.matmul(self.Wa, inputs_trans) + self.bias
        at = tf.matmul(self.Wa, inputs_trans)
        # Softmax
        # at = K.exp(at - K.max(at, axis=-1, keepdims=True))
        if mask is not None:
            # print(mask.shape)
            mask = tf.expand_dims(mask, axis=1)
            # print(mask.shape)
            mask = tf.tile(mask, [1, self.class_num, 1])
            # print(mask.shape)
            # print(type(mask))
            mask = tf.cast(mask, K.floatx())
            at += mask * -1e12
        # attention_weights = at / K.sum(at, axis=-1, keepdims=True)
        attention_weights = K.softmax(at, axis=-1)
        # print(attention_weights)
        # weighted sum
        # v = (batch_size, the number of classes, the number of LSTM dims)
        attention_adjusted_output = K.batch_dot(attention_weights, inputs)

        return attention_weights, attention_adjusted_output

    def get_config(self):
        config = super(LSTM_Attention, self).get_config()
        config['class_num'] = self.class_num
        return config
'''
'''
class CNN_Attention(Layer):

    def __init__(self, class_num, **kwargs):
        super(CNN_Attention, self).__init__(**kwargs)
        self.class_num = class_num

    def build(self, input_shape):
        self.filter_num = input_shape[2]

        # self.Wa = (the number of classes, the number of filters)
        self.Wa = self.add_weight(shape=(self.class_num, self.filter_num),
                                  initializer=initializers.get('glorot_uniform'),
                                  trainable=True,
                                  name='weights')

        super(CNN_Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):

        return None

    def call(self, inputs, mask=None):
        # inputs_trans = (batch_size, the number of filters, sentence_length)
        inputs_trans = tf.transpose(inputs, [0, 2, 1])

        # at = (batch_size, the number of classes, sentence_length)
        at = tf.matmul(self.Wa, inputs_trans)
        # dk = tf.cast(tf.shape(inputs)[-1], tf.float32)
        # scaled_attention_logits = at / tf.math.sqrt(dk)  # scale by sqrt(dk)
        attention_weights = K.softmax(at, axis=-1)
        # Softmax
        # at = K.exp(at - K.max(at, axis=-1, keepdims=True))
        # if mask is not None:
        #     at = at - (1 - mask) * self.padding_num

        # attention_weights = at / K.sum(at, axis=-1, keepdims=True)
        # print(attention_weights)
        # weighted sum
        # v = (batch_size, the number of classes, the number of filters)
        attention_adjusted_output = K.batch_dot(attention_weights, inputs)

        return attention_weights, attention_adjusted_output

    def get_config(self):
        config = super(CNN_Attention, self).get_config()
        config['class_num'] = self.class_num
        return config
'''