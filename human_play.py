# -*- coding: utf-8 -*-

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_keras import PolicyValueNet  # Keras


class Human(object):
    """
    人类玩家
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8,8
    model_file = 'best_policy.hdf5'
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)

    #读取训练好的模型
    best_policy = PolicyValueNet(width, height, model_file = model_file)
    # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
    mcts_player = MCTS_Pure(c_puct=5, n_playout=5000)#纯MCTS

    human = Human()

    # 把人设置为0为先手
    game.start_play(human, mcts_player, start_player=1, is_shown=1)



if __name__ == '__main__':
    run()
