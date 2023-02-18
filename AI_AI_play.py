from mcts import *
from tic_tac_toe import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def AI_AI_game(n1, n2):
    state = TicTacToeState()
    player_1, player_2 = 1, -1
    current_player = 1
    while not state.is_game_over():
        #state.display_board()
        #print()
        if current_player == player_1:
            move = get_mcts_move(state, player_1, n1)
        else:
            move = get_mcts_move(state, player_2, n2)
        state.play_move(move, current_player)
        current_player *= -1
    #state.display_board()
    result = state.get_result()
    if result == player_1:
        return "player_1", state
    elif result == player_2:
        return "player_2", state
    else:
        return "draw", state

def statistical_table(number_of_games, n1, n2):
    table_results = {"player_1":0, "player_2":0, "draw":0}
    for i in range(number_of_games):
        result_game, final_state = AI_AI_game(n1, n2)
        final_state.display_board()
        table_results[result_game] += 1
    return pd.Series(table_results)

table_results = statistical_table(10, 1000, 100)
print(table_results)
table_results.plot(kind = "bar")