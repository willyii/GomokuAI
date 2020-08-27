# -*- coding: utf-8 -*-

from __future__ import print_function

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras import models

# from tensorflow.keras.utils import np_utils

import numpy as np
import pickle


class PolicyValueNet():

    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height 
        self.l2_const = 1e-4
        self.create_policy_value_net()   
        self._loss_train_op()

        if model_file:
            self.model = models.load_model(model_file)
        
    def create_policy_value_net(self):
        in_x = network = Input((4, self.board_width, self.board_height))

        # 卷积层
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # 行动策略
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width*self.board_height, activation="softmax", kernel_regularizer=l2(self.l2_const))(policy_net)
        # 状态价值
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = Model(in_x, [self.policy_net, self.value_net])
        
        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results
        self.policy_value = policy_value
        
    def policy_value_fn(self, board):
        """
        输入为棋盘的状态，输出为行动和对应的概率，以及状态得分
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        有三个部分组成loss，
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        self.train_step = train_step

    def save_model(self, model_file):
        """ 存储模型"""
        self.model.save(model_file)
