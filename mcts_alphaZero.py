# -*- coding: utf-8 -*-


import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    定义MCTS中的树节点
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """
        扩展
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        根据值，选择
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        更新值
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """
        递归更新
        """

        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        计算并且返回当前节点的值，结合了一系列参数
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
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
 
            action, node = node.select(self._c_puct)
            state.do_move(action)

        #通过神经网络来评估叶子节点，输出行动和概率，同时输出当前玩家视角下的评分
        action_probs, leaf_value = self._policy(state)

        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:

            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 更新
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        运行所有的模拟过程，返回可行的行动以及其对应的概率，temp是温度参数位于0，1，用来平衡探索和利用
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)


        #计算行动概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

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

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables

        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 加入噪声
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新跟节点
                self.mcts.update_with_move(move)
            else:
                # 选择最高概率的落子
                move = np.random.choice(acts, p=probs)
                # 重设树
                self.mcts.update_with_move(-1)
                location = board.move_to_location(move)
                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
