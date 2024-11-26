import tkinter as tk
from genetic_algorithm import genetic_algorithm
from fitness_function import print_board, network_move
from minimax import minimax_move, check_winner, is_draw
from neural_network import NeuralNetwork

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jogo da Velha - IA vs Minimax")
        self.root.geometry("400x500")  # Define o tamanho da janela
        
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.current_state = [["" for _ in range(3)] for _ in range(3)]
        self.player_turn = "user"
        
        self.network = None
        self.minimax_level = "medium"
        self.create_widgets()
        self.initialize_network()

    def create_widgets(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.root, text="", font=("Arial", 24), width=5, height=2,
                                  command=lambda row=i, col=j: self.on_click(row, col))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

        self.status_label = tk.Label(self.root, text="Treinando a rede neural...", font=("Arial", 14))
        self.status_label.grid(row=3, column=0, columnspan=3)

        self.play_minimax_button = tk.Button(self.root, text="Jogar contra Minimax", command=self.play_with_minimax)
        self.play_minimax_button.grid(row=4, column=0, columnspan=1, pady=10)

        self.train_network_button = tk.Button(self.root, text="Treinar Rede com Minimax", command=self.train_network)
        self.train_network_button.grid(row=4, column=1, columnspan=1, pady=10)

        self.play_network_button = tk.Button(self.root, text="Jogar contra Rede Treinada", command=self.play_with_network)
        self.play_network_button.grid(row=4, column=2, columnspan=1, pady=10)

    def initialize_network(self):
        input_size = 9
        hidden_size = 64
        output_size = 9
        total_weights = input_size * hidden_size + hidden_size * output_size

        initial_game_state = [["", "", ""], ["", "", ""], ["", "", ""]]

        best_weights = genetic_algorithm(
            pop_size=5,
            weight_shape=total_weights,
            generations=100,
            game_state=initial_game_state,
            minimax_level=self.minimax_level
        )

        self.network = NeuralNetwork(input_size, hidden_size, output_size, weights=best_weights)
        self.status_label.config(text="Rede neural treinada! Você pode jogar agora.")

    def on_click(self, row, col):
        if self.current_state[row][col] == "" and self.player_turn == "user":
            self.current_state[row][col] = "O"
            self.buttons[row][col].config(text="O")
            self.player_turn = "minimax"
            self.check_game_status()
            if self.player_turn == "minimax":
                self.minimax_play()

    def minimax_play(self):
        move = minimax_move(self.current_state, self.minimax_level)
        if move is not None:
            row, col = move
            self.current_state[row][col] = "X"
            self.buttons[row][col].config(text="X")
            self.player_turn = "user"
            self.check_game_status()

    def network_play(self):
        move = network_move(self.network, self.current_state)
        if move is not None:
            row, col = move
            self.current_state[row][col] = "X"
            self.buttons[row][col].config(text="X")
            self.player_turn = "user"
            self.check_game_status()

    def check_game_status(self):
        winner = check_winner(self.current_state)
        if winner:
            self.status_label.config(text=f"{winner} venceu!")
            self.disable_buttons()
        elif is_draw(self.current_state):
            self.status_label.config(text="Empate!")
            self.disable_buttons()

    def disable_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(state=tk.DISABLED)

    def reset_board(self):
        for i in range(3):
            for j in range(3):
                self.current_state[i][j] = ""
                self.buttons[i][j].config(text="", state=tk.NORMAL)
        self.player_turn = "user"

    def play_with_minimax(self):
        self.reset_board()
        self.player_turn = "user"
        self.status_label.config(text="Você está jogando contra o Minimax")

    def train_network(self):
        self.initialize_network()
        self.status_label.config(text="Rede neural treinada! Você pode jogar agora.")

    def play_with_network(self):
        self.reset_board()
        self.player_turn = "user"
        self.status_label.config(text="Você está jogando contra a Rede Treinada")

if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()
