import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tic_tac_toe import TicTacToeState, display_board
length_side = 3
state_size = length_side ** 2
nh1, nh2 = 64, 64
np1, np2 = 64, 64

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, nh1)
        self.fc2 = nn.Linear(nh1, nh2)
        self.fc3 = nn.Linear(nh2, 1)
    
    def forward(self, x):
        y= torch.LongTensor(x)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.tanh(self.fc3(y))
        return y
    
class PolicyNetwork(nn.Module):
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, np1)
        self.fc2 = nn.Linear(np1, np2)
        self.fc3 = nn.Linear(np2, state_size)
    
    def forward(self, x):
        y= torch.LongTensor(x)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y))
        return y

def generate_board(state_size):
    number_of_turns = np.random.randint(np.floor(state_size / 2))
    res = [0 for i in range(state_size)]
    for i in range(number_of_turns):
        j = np.random.randint(state_size)
        while res[j] != 0:
            j = np.random.randint(state_size)
        res[j] = 1
        while res[j] != 0:
            j = np.random.randint(state_size)
        res[j] = -1
    return list(res)


def generate_new_board(boards_already_explored, state_size):
    new_board = generate_board(state_size)
    max_iterations = np.math.factorial(state_size)
    for i in range(max_iterations):
        if str(new_board) not in boards_already_explored:
            break
        new_board = generate_board(state_size)
    boards_already_explored.add(str(boards_already_explored))
    return new_board

def compute_outcome(starting_board, policy_network):
    current_state = TicTacToeState(starting_board)
    while not current_state.is_game_over():
        policy = policy_network.forward(current_state.board)
        move = np.argmax(np.random.multinomial(100, policy))
        current_state.play_move(move)
    return current_state.get_result()

def generate_dataset_policy_evaluation(policy_network, state_size, size_dataset):
    X, Y = [], []
    boards_already_explored = set()
    for i in range(size_dataset):
        x = generate_new_board(boards_already_explored, state_size)
        y = compute_outcome(x, policy_network)
        X.append(x)
        y.append(y)

value_network = ValueNetwork()
optimizer_value = optim.Adam(value_network.parameters(), lr = 0.01)
criterion_value = nn.MSELoss()

policy_network = PolicyNetwork()
optimizer_policy = optim.Adam(policy_network.parameters(), lr = 0.01)
criterion_policy = nn.CrossEntropyLoss()

generate_dataset_policy_evaluation(policy_network, state_size, 1)