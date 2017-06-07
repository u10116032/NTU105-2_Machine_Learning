# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:07:22 2017

@author: halley
"""

# CFModel.py


import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense, Activation
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential

class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,), name='movie_embedded'))
        P_bias = Sequential()
        P_bias.add(Embedding(n_users, 1, input_length=1))
        P_bias.add(Reshape((1,)))
        Q_bias = Sequential()
        Q_bias.add(Embedding(m_items, 1, input_length=1))
        Q_bias.add(Reshape((1,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([Merge([P, Q], mode='dot', dot_axes=1), P_bias, Q_bias], mode='sum'))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id]), np.array([user_id]), np.array([item_id])])[0][0]

class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))       
        P_bias = Sequential()
        P_bias.add(Embedding(n_users, 1, input_length=1))
        P_bias.add(Reshape((1,)))
        Q_bias = Sequential()
        Q_bias.add(Embedding(m_items, 1, input_length=1))
        Q_bias.add(Reshape((1,)))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q, P_bias, Q_bias], mode='concat'))
        self.add(Dropout(p_dropout))
        self.add(Dense(k_factors+2))
        self.add(Activation(PReLU()))
        self.add(Dropout(p_dropout))
        self.add(Dense(1, activation='linear'))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id]), np.array([user_id]), np.array([item_id])])[0][0]