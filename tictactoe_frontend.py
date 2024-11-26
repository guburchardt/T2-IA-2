import numpy as np
from genetic_algorithm import genetic_algorithm
from fitness_function import network_move
from minimax import minimax_move, check_winner, is_draw
from neural_network import NeuralNetwork

def print_board(board):
    for row in board:
        print(" | ".join([cell if cell != "" else " " for cell in row]))
        print("-" * 5)

def user_vs_minimax():
    current_state = [["" for _ in range(3)] for _ in range(3)]
    player_turn = "user"
    minimax_level = "medium"

    while True:
        print_board(current_state)
        if player_turn == "user":
            move = get_user_move(current_state)
            current_state[move[0]][move[1]] = "O"
            player_turn = "minimax"
        else:
            move = minimax_move(current_state, minimax_level)
            if move is not None:
                current_state[move[0]][move[1]] = "X"
            player_turn = "user"

        winner = check_winner(current_state)
        if winner:
            print_board(current_state)
            print(f"{winner} venceu!")
            break
        elif is_draw(current_state):
            print_board(current_state)
            print("Empate!")
            break

def train_network():
    print("Treinando a Rede Neural... Pressione 'Enter' a qualquer momento para cancelar o treinamento.")
    input_size = 9
    hidden_size = 64
    output_size = 9
    total_weights = input_size * hidden_size + hidden_size * output_size

    initial_game_state = [["", "", ""], ["", "", ""], ["", "", ""]]

    try:
        best_weights = genetic_algorithm(
            pop_size=5,
            weight_shape=total_weights,
            generations=100,
            game_state=initial_game_state,
            minimax_level="medium"
        )
    except KeyboardInterrupt:
        print("Treinamento cancelado.")
        return None

    return NeuralNetwork(input_size, hidden_size, output_size, weights=best_weights)

def user_vs_trained_network(network):
    current_state = [["" for _ in range(3)] for _ in range(3)]
    player_turn = "user"

    while True:
        print_board(current_state)
        if player_turn == "user":
            move = get_user_move(current_state)
            current_state[move[0]][move[1]] = "O"
            player_turn = "network"
        else:
            move = network_move(network, current_state)
            if move is not None:
                current_state[move[0]][move[1]] = "X"
            player_turn = "user"

        winner = check_winner(current_state)
        if winner:
            print_board(current_state)
            print(f"{winner} venceu!")
            break
        elif is_draw(current_state):
            print_board(current_state)
            print("Empate!")
            break

def get_user_move(board):
    while True:
        try:
            move = int(input("Digite sua jogada (1-9): ")) - 1
            row, col = divmod(move, 3)
            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == "":
                return row, col
            else:
                print("Movimento inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número entre 1 e 9.")

if __name__ == "__main__":
    trained_network = None
    while True:
        print("Escolha uma opção:")
        print("1. Jogar contra o Minimax")
        print("2. Treinar Rede Neural")
        print("3. Jogar contra a Rede Treinada")
        print("4. Sair")
        choice = input("Opção: ")

        if choice == "1":
            user_vs_minimax()
        elif choice == "2":
            trained_network = train_network()
            if trained_network:
                print("Rede Neural treinada com sucesso!")
            else:
                print("Treinamento foi cancelado.")
        elif choice == "3":
            if trained_network:
                user_vs_trained_network(trained_network)
            else:
                print("A rede neural ainda não foi treinada. Treine-a primeiro.")
        elif choice == "4":
            break
        else:
            print("Opção inválida. Tente novamente.")
