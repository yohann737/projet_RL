from tic_tac_toe import TicTacToeState, display_board
from mcts import *
random.seed(42)
def get_human_move(board):
    while True:
        move = int(input('Enter your move (0-8): '))
        if move in board.get_possible_moves():
            return move
        print('Invalid move')

def play_game(root):
    state = root.state
    while not state.is_game_over():
        print(display_board(state.board))
        print()
        if state.player == 1:
            move = get_human_move(state)
        else:
            mcts(root, iterations = 1000)
            node = select_child(root)
            move = node.move
        state.play_move(move)
        root = Node(state = state)
    print(display_board(state.board))
    result = state.get_result()
    if result == 1:
        print('You win!')
    elif result == -1:
        print('You lose!')
    else:
        print('Draw!')

state = TicTacToeState()
root = Node(state)
play_game(root)