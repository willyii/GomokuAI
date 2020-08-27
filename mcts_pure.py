# -*- coding: utf-8 -*-

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    # 随机策略
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """
    #将棋盘状态作为输入，输出一个行动以及其对应的概率，再加上当前状态的评分
    """
    # 返回均匀概率，并且对当前状态评分为0
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """
    MCTS中的节点，每个节点都记录它自己的值Q，先验概率P，以及访问次数调整的得分u
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 存储每次行动到节点的对应
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        通过创造子节点来扩展树，action_pirors：一系列可行行动以及其对应的概率，根据策略函数得出
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        选择子节点中具有最大值对应的行动，返回行动以及其对应的转换的节点
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        从叶子节点开始更新每个节点的值，leaf_value：从当前玩家角度的子数的值
        """
        # 计算访问次数
        self._n_visits += 1
        # 计算Q值
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        递归更新，与update相似，只不过是递归的更细父节点
        """
        # 如果不是跟节点，则该节点的父节点先更新
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算并且返回该节点的值，该值结合了子数评估值Q，以及该节点的访问次数等参数，c_puct参数调整平衡
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """
        判断该节点是否为叶子节点
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn：一个函数，该函数以当前棋盘的状态为输入，输出每个行动以及其对应的概率，同样会以当前玩家的视角对当前棋盘状态进行打分
        c_puct: 同上，一个控制的参数
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        进行一次简单的模拟，从根到叶节点，得到叶子节点的值，再反向传播更新其父母的值
        """
        node = self._root
        while(1):
            if node.is_leaf():

                break
            # 选择下一步落子
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # 判断是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 通过随机策略评估叶子节点
        leaf_value = self._evaluate_rollout(state)
        # 更新值
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """
        通过随机模拟来进行对局知道游戏结束，如果当前玩家胜利，返回1，否则返回-1，平局返回0
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        if winner == -1:  #平局
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """
        运行所有模拟，然后返回访问次数最多的落子
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """
        对树木访问下一个节点，保存信息，当传入-1时，是重设树
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """
    基于MCTS的AI玩家
    """
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
