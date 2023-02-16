from mcts import *
from tic_tac_toe import *

def get_human_move(current_player):
    # Define player symbols
    symbols = {1: "X", -1: "O"}
    move = None
    while move is None:
        try:
            move = int(input("Player {}'s turn. Enter a move (0-8): ".format(symbols[current_player])))
            if move not in state.get_possible_moves():
                print("That space is already taken. Please choose another move.")
                move = None
            else:
                state.play_move(move, current_player)
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 8.")
    return move
def human_game():
    # Initialize the game state
    state = TicTacToeState()
    # Define the current player
    current_player = 1

    # Print the initial board
    state.display_board()
    # Loop until the game is over
    while not state.is_game_over():
        # Get the current player's move
        move = get_human_move(current_player)
        state.play_move(move, current_player)
        # Switch to the other player
        current_player *= -1
        # Print the updated board
        state.display_board()
    # Print the final result
    result = state.get_result()
    if result == -1:
        print("Game over! -1 wins!")
    elif result == 1:
        print("Game over! 1 wins!")
    elif result == 0:
        print("Game over! It's a draw!")

def play_mcts():
    state = TicTacToeState()
    root = Node(state, 1)
    human_player = 1
    ai_player = -1 * human_player
    current_player = 1
    while not state.is_game_over():
        state.display_board()
        print()
        if current_player == human_player:
            move = get_human_move(current_player)
        else:
            root = Node(state, ai_player)
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
        state.play_move(move, current_player)
        current_player *= -1
    state.display_board()
    result = state.get_result()
    if result == human_player:
        print('You win!')
    elif result == ai_player:
        print('You lose!')
    else:
        print('Draw!')

play_mcts()