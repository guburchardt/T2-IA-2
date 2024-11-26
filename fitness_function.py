from minimax import minimax_move, check_winner, is_draw
import numpy as np
from neural_network import NeuralNetwork

def print_board(state):
    for row in state:
        print(" | ".join(row))
        print("-" * 9)

def fitness_function(weights, game_state, minimax_level):
    network = NeuralNetwork(input_size=9, hidden_size=64, output_size=9, weights=weights)
    initial_state = game_state
    invalid_moves_count = 0
    player_turn = "network"
    draw_bonus = 10 

    while True:
        current_state = [row[:] for row in initial_state]
        
        while True:
            winner = check_winner(current_state)
            if winner == "network":
                print("Rede neural venceu!")
                print_board(current_state)
                return 20 - invalid_moves_count 
            elif winner == "minimax":
                print("Minimax venceu!")
                print_board(current_state)
                return -10 - invalid_moves_count 
            elif is_draw(current_state):
                print("Empate!")
                print_board(current_state)
                return draw_bonus - invalid_moves_count 

            if player_turn == "network":
                move = network_move(network, current_state)
                if move is None:
                    print("Movimento inválido pela rede neural!")
                    return -20
                current_state[move[0]][move[1]] = "X"
                player_turn = "minimax"
            else:
                move = minimax_move(current_state, minimax_level)
                if move is None:
                    print("Movimento inválido pelo minimax!")
                    return -20
                current_state[move[0]][move[1]] = "O"
                player_turn = "network"

def is_valid_move(board, move):
    row, col = move
    return 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ""

def network_move(network, board):
    flat_board = np.array([1 if cell == "X" else -1 if cell == "O" else 0 for row in board for cell in row])

    output = network.forward(flat_board)

    move_indices = np.argsort(output)[::-1] 

    for move in move_indices:
        row, col = divmod(move, 3)
        if is_valid_move(board, (row, col)):
            return (row, col)
          
    return None
