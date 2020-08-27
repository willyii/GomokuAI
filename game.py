import numpy as np
import tkinter
import math

class Board(object):

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # 棋盘状态记录为字典模式，其中键为落子的位置，对应的值为落子的玩家
        self.states = {} 
        # 判断胜利条件，当几个子连起来的时候，可以达到胜利
        self.n_in_row = int(kwargs.get('n_in_row', 5)) 
        self.players = [1, 2] 
        
    def init_board(self, start_player=0):
        '''
        初始化棋盘
        '''
        self.current_player = self.players[start_player]  #先落子的玩家
        #保存可以落子的位置
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        '''
        落子，此处的将落子move转化为落子坐标：
        '''
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        '''
        上面函数的返过程，将坐标转化为move
        '''
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        '''
        返回当前玩家视角下的棋盘状态，该状态包含了前四步落子情况和当前棋盘情况，所以对应的shape为
        4*棋盘宽*棋盘长
        '''

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 指出上次落子的位置
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # 指出当前落子的颜色
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if(len(moved) < self.n_in_row*2 - 1):
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        '''
        判断游戏是否结束，返回胜利玩家的id
        '''
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

class Point:

    def __init__(self, x, y):
        self.x = x;
        self.y = y;
        self.pixel_x = 30 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y

class Game(object):
    def __init__(self, board, **kwargs):
        self.board = board
    
    def click1(self, event): #点鼠标
        size = self.board.width
        current_player = self.board.get_current_player()
        if current_player == 1:
            for i in range(size):
                for j in range(size):
                    move = self.board.location_to_move((i, j))
                    if move in self.board.availables:
                        square_distance = math.pow((event.x - self.chess_board_points[i][j].pixel_x), 2) + math.pow((event.y - self.chess_board_points[i][j].pixel_y), 2)                        
                        
                        if (square_distance <= 200):
                            self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='black')
                            self.board.do_move(move)
    
    def run(self):
        current_player = self.board.get_current_player()
        
        end, winner = self.board.game_end()
        if end:
            if winner != -1:
                self.cv.create_text(140, 270, text="Game over. Winner is {}".format(self.players[winner]))
                self.cv.unbind('<Button-1>')
            else:
                self.cv.create_text(140, 270, text="Game end. Tie")

            return winner
        else:
            self.cv.after(100, self.run)

        if current_player == 2:
            player_in_turn = self.players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            i, j = self.board.move_to_location(move)
            self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='white')
                
        
    def graphic(self, board, player1, player2):
        '''
        画棋盘啦
        '''
        width = board.width
        height = board.height
        
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        self.players = {p1: player1, p2:player2}
        
        window = tkinter.Tk()
        self.cv = tkinter.Canvas(window, height=300, width=280, bg = 'white')
        self.chess_board_points = [[None for i in range(width)] for j in range(height)]
        
        for i in range(width):
            for j in range(height):
                self.chess_board_points[i][j] = Point(i, j);
        for i in range(width):  #竖线
            self.cv.create_line(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y, self.chess_board_points[i][width-1].pixel_x, self.chess_board_points[i][width-1].pixel_y)
        
        for j in range(height):  #横线
            self.cv.create_line(self.chess_board_points[0][j].pixel_x, self.chess_board_points[0][j].pixel_y, self.chess_board_points[height-1][j].pixel_x, self.chess_board_points[height-1][j].pixel_y)        
        
        self.button = tkinter.Button(window, text="start game!", command=self.run)
        self.cv.bind('<Button-1>', self.click1)
        self.cv.pack()
        self.button.pack()
        window.mainloop()
               
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        '''
        开玩
        '''
        self.board.init_board(start_player)

        if is_shown:
            self.graphic(self.board, player1, player2)
        else:
            p1, p2 = self.board.players
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2:player2}
            while(1):
                current_player = self.board.get_current_player()
                player_in_turn = players[current_player]
                move = player_in_turn.get_action(self.board)
                self.board.do_move(move)
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)
                end, winner = self.board.game_end()
                if end:
                    return winner   

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        ''' 
        使用MCTS进行自我对局，并且存下自我对局的数据(state, mcts_probs, z)用作后来的训练数据
        '''
        self.board.init_board()        
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []        
        while(1):
            move, move_probs = player.get_action(self.board, 
                                                 temp=temp, 
                                                 return_prob=1)
            # 存储数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # 进行落子                      
            self.board.do_move(move)      
            end, winner = self.board.game_end()
            if end:
                #当前玩家视角下，每个状态下的胜利玩家
                winners_z = np.zeros(len(current_players))  
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MCTS跟节点
                player.reset_player() 
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            