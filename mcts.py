import random
import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

def select_child(node):
    total_visits = sum(child.visits for child in node.children)
    log_visits = np.log(1 + total_visits)

    def uct(node):
        return node.wins / (1 + node.visits) + 2 * np.sqrt(log_visits / (node.visits + 1))

    return max(node.children, key=uct)

def expand(node):
    state = node.state.copy()
    move = random.choice(state.get_possible_moves())
    #print("move", move)
    child_state = state.copy()
    child_state.play_move(move)
    child_node = node.add_child(child_state)
    return child_node
    
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
        if not node.state.is_game_over():
            child_node = expand(node)
        else:
            child_node = node
        result = simulate(child_node)
        backpropagate(child_node, result)
    for child in root.children:
        print(child.visits)