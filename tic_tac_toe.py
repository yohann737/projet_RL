import numpy as np
import torch
import torch.nn as nn
import random
import pdb

class TicTacToeState:
    def __init__(self):
        self.board = [0] * 9
    
    def play_move(self, move, player):
        if player not in [-1, 1] or move not in self.get_possible_moves():
            raise ValueError("Player not in [-1, 1] or the move in invalid")
        self.board[move] = player
    def get_possible_moves(self):
        res = []
        for i in range(len(self.board)):
            if self.board[i] == 0:
                res.append(i)
        return res
    def play_random_move(self, player):
        self.play_move(random.choice(self.get_possible_moves()), player)
    
    def display_board(self):
        symbols = {1: "X", -1: "O", 0: " "}
        print(" " + symbols[self.board[0]] + " | " + symbols[self.board[1]] + " | " + symbols[self.board[2]] + " ")
        print("-----------")
        print(" " + symbols[self.board[3]] + " | " + symbols[self.board[4]] + " | " + symbols[self.board[5]] + " ")
        print("-----------")
        print(" " + symbols[self.board[6]] + " | " + symbols[self.board[7]] + " | " + symbols[self.board[8]] + " ")

    def get_result(self):
        # check for rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] and self.board[i] != 0:
                return self.board[i]
        # check for columns
        for i in range(0, 3):
            if self.board[i] == self.board[i+3] == self.board[i+6] and self.board[i] != 0:
                return self.board[i]
        # check for diagonals
        if self.board[0] == self.board[4] == self.board[8] and self.board[0] != 0:
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] and self.board[2] != 0:
            return self.board[2]
        # check for draw
        if 0 not in self.board:
            return 0
        # game not over
        return None
    def is_game_over(self):
        return self.get_result() != None


