import random
import numpy as np
from tic_tac_toe import display_board
class Node:
    def __init__(self, state, parent=None, move = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state, move):
        child = Node(child_state, self, move = move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result
    
    def uct(self):
        #total_visits = sum(child.visits for child in self.children)
        #log_visits = np.log(total_visits)
        numerator_exploration = 15 * np.log(self.parent.visits)
        denominator_exploration = self.visits
        if self.visits != 0:
            return self.wins / self.visits + np.sqrt(numerator_exploration / denominator_exploration)
        else:
            return np.float('inf')
        
def select_child(node):
    maximum, res = node.children[0].uct(), node.children[0]
    for child in node.children:
        score = child.uct()
        if score > maximum:
            maximum = score
            res = child
    return res

def expand(node):
    state = node.state.copy()
    for move in state.get_possible_moves():
        child_state = state.copy()
        child_state.play_move(move)
        node.add_child(child_state, move)
    return random.choice(node.children)
    
def simulate(node):
    state = node.state.copy()
    while not state.is_game_over():
        state.play_random_move()
    return state.get_result()

def backpropagate(node, result):
    while node is not None:
        node.update(result)
        node = node.parent

def mcts(root, iterations):
    for i in range(iterations):
        node = root
        while len(node.children) != 0 and not node.state.is_game_over():
            node = select_child(node)
        if node.visits == 0 or node.state.is_game_over():
            child_node = node
        else:
            child_node = expand(node)
        result = simulate(child_node)
        backpropagate(child_node, result)
    for child in root.children:
        print(child.move, child.visits, child.wins, child.uct())
        print(display_board(child.state.board))