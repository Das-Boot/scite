# -*- coding: utf-8 -*-

'''
Modified from: Zhao HG @ https://github.com/CyberZHG/keras-multi-head/blob/master/keras_multi_head/multi_head_attention.py
'''

import keras
import keras.backend as K
from keras import regularizers
from keras import constraints
from keras import initializers
from keras_self_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(keras.layers.Layer):
    """Multi-head self attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 size_per_head,
                 activation='relu',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param size_per_head: Size of per head
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'size_per_head': self.size_per_head,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.head_num*self.size_per_head)

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        q = k = v = input_shape
        feature_dim = self.head_num*self.size_per_head
        self.Wq = self.add_weight(
            shape=(q[-1], feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(k[-1], feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(v[-1], feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadSelfAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num, size_per_head):
        input_shape = K.shape(x)
        batch_size, seq_len = input_shape[0], input_shape[1]
        x = K.reshape(x, (batch_size, seq_len, head_num, size_per_head))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, size_per_head))

    @staticmethod
    def _reshape_from_batches(x, head_num, size_per_head):
        input_shape = K.shape(x)
        batch_size, seq_len = input_shape[0], input_shape[1]
        x = K.reshape(x, (batch_size // head_num,
                          head_num, seq_len, size_per_head))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, head_num*size_per_head))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, K.stack([1, head_num, 1]))
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num, self.size_per_head),
                self._reshape_to_batches(k, self.head_num, self.size_per_head),
                self._reshape_to_batches(v, self.head_num, self.size_per_head),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num, self.size_per_head)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        return y