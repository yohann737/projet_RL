import random
class TicTacToeState:
    def __init__(self, board=None, player=1):
        if board is None:
            board = [0] * 9
        self.board = board
        self.player = player

    def get_possible_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def play_move(self, move):
        self.board[move] = self.player
        self.player = -self.player

    def play_random_move(self):
        possible_moves = self.get_possible_moves()
        move = random.choice(possible_moves)
        self.play_move(move)

    def is_game_over(self):
        return self.get_result() is not None

    def get_result(self):
        for i in range(3):
            row = i * 3
            if self.board[row] != 0 and self.board[row] == self.board[row + 1] == self.board[row + 2]:
                return self.board[row]
            if self.board[i] != 0 and self.board[i] == self.board[i + 3] == self.board[i + 6]:
                return self.board[i]

        if self.board[0] != 0 and self.board[0] == self.board[4] == self.board[8]:
            return self.board[0]
        if self.board[2] != 0 and self.board[2] == self.board[4] == self.board[6]:
            return self.board[2]

        if 0 not in self.board:
            return 0

        return None

    def copy(self):
        return TicTacToeState(self.board[:], self.player)
def display_board(board):
    symbols = [' ', 'X', 'O']
    rows = [' {} | {} | {}'.format(*[symbols[v] for v in board[i:i + 3]])
            for i in range(0, 9, 3)]
    separator = '\n---+---+---\n'
    return separator.join(rows)