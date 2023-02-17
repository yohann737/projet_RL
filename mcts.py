import random
import numpy as np
import copy
import tic_tac_toe as ttt
import pdb
class Node:
    def __init__(self, state, player, parent = None):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
    
    def ucb(self):
        if self.visits == 0:
            return np.float("inf")
        exploitation = self.player * self.wins / self.visits
        numerator_exploration = np.log(self.parent.visits)
        denominator_exploration = self.visits
        if denominator_exploration == 0:
            return np.float("inf")
        else:
            return exploitation + 2 * np.sqrt(numerator_exploration / denominator_exploration)
    
    def select_child(self):
            score_ucb, res = self.children[0].ucb(), self.children[0]
            for child in self.children:
                if child.ucb() > score_ucb:
                    score_ucb = child.ucb()
                    res = child
            return res
    
    def expand(self):
        possible_moves = self.state.get_possible_moves()
        for move in possible_moves:
             state_copy = copy.deepcopy(self.state)
             state_copy.play_move(move, self.player)
             child = Node(state_copy, -1 * self.player, self)
             self.children.append(child)
        return random.choice(self.children)
    
    def simulate(self):
        state_copy = copy.deepcopy(self.state)
        current_player = self.player
        while not state_copy.is_game_over():
            #state_copy.display_board()
            state_copy.play_random_move(current_player)
            current_player *= -1
        return state_copy.get_result()# A verifier
    
    def backpropagate(self, result):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
    
def mcts(root, n):
    for i in range(n):
        node = root
        while len(node.children) != 0 and not node.state.is_game_over():
            node = node.select_child()
        if not node.state.is_game_over():
            child_node = node.expand()
        else:
            child_node = node
        result = root.player * child_node.simulate()
        child_node.backpropagate(result)


def get_mcts_move(state, player):
    root = Node(state, player)
    mcts(root, 1000)
    node = root.children[0]
    max_score = node.wins / node.visits
    for child in root.children:
        score = child.wins / child.visits
        if score > max_score:
            max_score = score
            node = child
    state_difference = [np.abs(root.state.board[i] - node.state.board[i]) for i in range(len(root.state.board))]
    move = np.argmax(state_difference)
    return move
    