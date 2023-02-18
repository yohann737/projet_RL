from mcts import *
from tic_tac_toe import *

def get_human_move(state, current_player):
    # Define player symbols
    symbols = {1: "X", -1: "O"}
    move = None
    while move is None:
        move = int(input("Player {}'s turn. Enter a move (0-8): ".format(symbols[current_player])))
        if move not in state.get_possible_moves():
            print("Invalid move. Please choose another move.")
            move = None
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

def play_mcts(human_player):
    state = TicTacToeState()
    ai_player = -1 * human_player
    current_player = 1
    while not state.is_game_over():
        state.display_board()
        print()
        if current_player == human_player:
            move = get_human_move(state, current_player)
            state.display_board()
            print(move)
        else:
            move = get_mcts_move(state, ai_player, 10**4)
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

try_again = True
while try_again:
    human_player = None
    while human_player == None:
        human_starts = input("Do you want to play first (Y/N)?")
        human_starts = human_starts[0].upper()
        if human_starts == "Y":
            human_player = 1
        elif human_starts == "N":
            human_player = -1
        else:
            print("Sorry, we did not understand your answer. Please answer using Y/N")
    play_mcts(human_player)
    try_again_str = input("Try again ? (Y/N)")
    try_again_str = try_again_str[0].upper()
    if try_again_str == "Y":
        try_again = True
    elif try_again_str == "N":
        try_again = False
    else:
        print("Sorry, we did not understand your answer. Please answer using Y/N")